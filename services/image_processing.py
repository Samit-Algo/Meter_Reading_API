import base64
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv
from groq import Groq
from io import BytesIO

load_dotenv()


class MeterService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

    async def groq_process_meter_image_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Process any type of meter image (electricity, water, gas, etc.) and return the numeric meter reading (e.g., kWh, cubic meters)
        along with a confidence percentage for that reading.
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

           

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are given an image of a utility meter. "
                                    "It may be an electricity meter (kWh), water meter, gas meter, or any other type. "
                                    "Analyze the image and extract ONLY the numeric meter reading as it appears "
                                    "(digits & decimal, include units if present, e.g., '37856.3 kWh', '08215 m³'). "
                                    "If units are visible, return them as part of the value. If not, just return the number. "
                                    "If the actual reading value is NOT visible or unclear, respond exactly with 'Not visible'. "
                                    "Do NOT interpret, summarize, or provide any extra commentary. Only output the exact value as described."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "meter_reading_only",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reading": {"type": ["string", "number", "null"]},
                            },
                            "required": ["reading"],
                        },
                    },
                },
            )

            response = chat_completion.choices[0].message.content.strip()
            data = json.loads(response)
            initial_reading = data.get("reading")

            return await self.validate_and_verify_meter_reading_with_confidence_percentage(image_bytes, initial_reading)
        except Exception as e:
            print(e)
            return {"reading": "Not visible", "confidence": "0%"}

    async def validate_and_verify_meter_reading_with_confidence_percentage(self, image_bytes: bytes, initial_reading: Any = None) -> Dict[str, Any]:
        """
        Validate and verify the meter reading from the image, and return the verified reading, confidence percentage (as a string, e.g. '98%'), 
        and if not visible, also include the reason.
        NOTE: The max confidence returned will be 97%.
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            validation_prompt = (
                "You are an expert in analyzing utility meter images. "
                "Your task is to carefully examine this meter image and validate the reading."
                "\n\nSTRICT INSTRUCTIONS:"
                "\n1. Check if the image is clear enough to read the meter display"
                "\n2. If the image is blurry, noisy, dark, or the meter display is not clearly visible, return 'Not visible'"
                "\n3. If the image is clear, extract the exact numeric meter reading (with units if visible)"
                "\n4. Verify the reading by double-checking all digits carefully"
                "\n5. Return ONLY the verified reading value (e.g., '37856.3 kWh', '08215', '1234.56'), your confidence IN PERCENTAGE (e.g., 97%, 83%),"
                " and IF the reading is 'Not visible', ALSO RETURN the reason why it is not visible (e.g. blurry, incomplete display, obscured)."
                "\n\nIMPORTANT:"
                "\n- If ANY digit is unclear or uncertain, return 'Not visible'"
                "\n- If the meter display is partially visible, return 'Not visible'"
                "\n- Only return a reading if you are HIGHLY CONFIDENT it is 100% accurate"
                "\n- Do not include any explanations, comments, or extra text except for the 'reason' if 'Not visible'"
                "\n- THE MAXIMUM CONFIDENCE YOU CAN RETURN IS 97% (DO NOT RETURN CONFIDENCE HIGHER THAN 97%)"
                "\n\nReturn JSON: {'final_reading': '', 'confidence': '', 'reason': ''} "
                "where confidence is a percentage as a string (e.g. '95%'), and reason is required ONLY if final_reading is 'Not visible'."
            )

            if initial_reading:
                validation_prompt += (
                    f"\n\nPrevious reading extracted: {initial_reading}. "
                    "Please verify if this is correct and provide a confidence (in percentage, max 97%). "
                    "If you return 'Not visible', also include a brief reason. "
                    "Output JSON: {'final_reading': <value>, 'confidence': <percentage string like \"97%\">, 'reason': <required if Not visible>}"
                )

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": validation_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "validated_reading_with_confidence_and_reason",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "final_reading": {"type": "string"},
                                "confidence": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["final_reading", "confidence"],
                        },
                    },
                },
            )
            response = chat_completion.choices[0].message.content.strip()
            data = json.loads(response)
            final_reading = data.get("final_reading", "Not visible")
            confidence = data.get("confidence", "0%")
            reason = data.get("reason", "")

            # Cap confidence at 97%
            def clamp_confidence(conf_str):
                if not conf_str or not conf_str.strip().endswith('%'):
                    return conf_str
                try:
                    value = int(conf_str.strip(' %'))
                    if value > 97:
                        value = 97
                    return f"{value}%"
                except Exception:
                    return conf_str

            # Build result dictionary
            result = {}
            if (
                final_reading and final_reading.strip().lower() != "null" and final_reading.strip() != ""
                and final_reading.strip().lower() != "not visible"
                and confidence and confidence.strip() != "" and confidence.strip().endswith('%')
            ):
                capped_confidence = clamp_confidence(confidence.strip())
                result = {"reading": final_reading.strip(), "confidence": capped_confidence}
            elif final_reading and final_reading.strip().lower() != "null" and final_reading.strip() != "" and final_reading.strip().lower() != "not visible":
                result = {"reading": final_reading.strip(), "confidence": "0%"}
            else:
                # It's Not visible
                not_visible_reason = reason.strip() if reason else "Not specified"
                result = {"reading": "Not visible", "confidence": "0%", "reason": not_visible_reason}
            return result

        except Exception as e:
            return {"reading": "Not visible", "confidence": "0%", "reason": "Internal error or exception"}

   