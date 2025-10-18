import base64
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv
from groq import Groq

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
                                    "You are an OCR extractor specialized in utility meters (electricity kWh, water m³, gas, etc.).\n\n"
                                    "Task:\n"
                                    "- Extract the exact meter register reading as displayed.\n"
                                    "- Include units if visible (e.g., '37856.3 kWh', '08215 m³'); otherwise return only the number.\n"
                                    "- Preserve leading zeros and the decimal/comma separator exactly as shown.\n\n"
                                    "Critically DO NOT return:\n"
                                    "- Serial numbers, model/firmware IDs, barcodes/QR codes, dates/timestamps, CT ratios, tariff codes, voltage/current/power values, 'max demand' or diagnostics.\n"
                                    "- Any text outside the main register display.\n\n"
                                    "If any digit is unclear, the display is obstructed/blurred, or the reading is not visible, return exactly 'Not visible'.\n\n"
                                    "Return format is JSON as instructed by the tool; do not add extra commentary."
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
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0,
                max_tokens=128,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "meter_reading_only",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "reading": {"type": ["string", "null"]},
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
                "You are validating a utility meter reading from an image.\n\n"
                "Instructions:\n"
                "1) If the main register reading is not fully legible or any digit is uncertain → final_reading = 'Not visible' and provide a brief reason.\n"
                "2) Otherwise, return the exact reading string as displayed (keep leading zeros, punctuation, and units if visible).\n"
                "3) Ignore serial numbers, dates, CT ratios, tariffs, diagnostic values, and anything outside the main register.\n\n"
                "Constraints:\n"
                "- confidence is a percentage string like '83%'. Maximum allowed is '97%'.\n"
                "- If final_reading == 'Not visible', reason is required.\n\n"
                "Return JSON only: {""final_reading"": ""..."", ""confidence"": ""..."", ""reason"": ""...""}"
            )

            if initial_reading:
                validation_prompt += (
                    f"\n\nPrevious reading extracted: {initial_reading}. "
                    "Please verify correctness and provide confidence (percentage string, max 97%). "
                    "If 'Not visible', include a brief reason."
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
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0,
                max_tokens=160,
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

   