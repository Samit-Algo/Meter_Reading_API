import base64
import io
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv
from groq import Groq
import boto3
from PIL import Image
import pillow_heif
from datetime import datetime
import asyncio
import random

load_dotenv()


class MeterService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        
        # AWS Bedrock client
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="ap-southeast-2")
        
        # Model configurations - AWS Model 1 is the primary model
        self.groq_model = "meta-llama/llama-4-maverick-17b-128e-instruct"

        # AWS Model IDs
        self.aws_model_1_id = "arn:aws:bedrock:ap-southeast-2:119071858069:inference-profile/au.anthropic.claude-haiku-4-5-20251001-v1:0"
        self.aws_model_2_id = "arn:aws:bedrock:ap-southeast-2:119071858069:inference-profile/apac.amazon.nova-pro-v1:0"

    # ============================================================================
    # IMAGE COMPRESSION
    # ============================================================================

    def _compress_like_whatsapp(self, image_bytes: bytes) -> bytes:
        """
        Compress image in a WhatsApp-like manner:
        - Decode HEIC/HEIF if needed
        - Convert to JPEG
        - Resize so the longer side <= 1280 px (maintain aspect ratio)
        - Use quality ~75 with subsampling for strong size reduction
        - Strip metadata
        Additionally, ensure the final JPEG is <= 250KB by adaptively lowering
        quality and, if necessary, further downscaling dimensions.
        """
        try:
            # Local helper to persist the output for download
            def _save_for_download(data: bytes) -> None:
                try:
                    downloads_dir = os.path.join(os.getcwd(), "compressed_downloads")
                    # os.makedirs(downloads_dir, exist_ok=True)
                    # filename = f"compressed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    # file_path = os.path.join(downloads_dir, filename)
                    # with open(file_path, "wb") as f:
                    #     f.write(data)
                except Exception:
                    # Silently ignore persistence errors to avoid impacting main flow
                    pass

            # Try HEIF decode first (iPhone HEIC)
            try:
                heif = pillow_heif.read_heif(image_bytes)
                img = Image.frombytes(heif.mode, heif.size, heif.data)
            except Exception:
                # Fallback to PIL open
                img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB for JPEG
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Initial resize: longest side to 1280px
            max_side = 1280
            w, h = img.size
            scale_to_max = min(1.0, max_side / float(max(w, h)))
            if scale_to_max < 1.0:
                new_size = (int(w * scale_to_max), int(h * scale_to_max))
                img = img.resize(new_size, Image.LANCZOS)

            # Enforce target size <= 250KB with adaptive quality/dimension reduction
            target_bytes = 250 * 1024
            min_quality = 35
            max_quality = 80
            quality_step = 5

            current_img = img
            best_bytes = None  # keep the smallest attempt in case target is unreachable
            long_side_min = 320  # avoid shrinking below this unless absolutely necessary

            while True:
                # Try qualities from high to low for the current dimensions
                q = max_quality
                while q >= min_quality:
                    out = io.BytesIO()
                    current_img.save(out, format="JPEG", quality=q, optimize=True, subsampling=2)
                    data = out.getvalue()
                    if len(data) <= target_bytes:
                        _save_for_download(data)
                        return data
                    if best_bytes is None or len(data) < len(best_bytes):
                        best_bytes = data
                    q -= quality_step

                # If still too big at min quality, downscale dimensions and retry
                w, h = current_img.size
                if max(w, h) <= long_side_min:
                    # Can't reasonably downscale further; return best effort
                    result_bytes = best_bytes if best_bytes is not None else image_bytes
                    _save_for_download(result_bytes)
                    return result_bytes

                # Reduce size modestly to preserve readability while shrinking bytes
                downscale_factor = 0.85 if max(w, h) > 640 else 0.9
                new_w = max(1, int(w * downscale_factor))
                new_h = max(1, int(h * downscale_factor))
                if (new_w, new_h) == (w, h):
                    result_bytes = best_bytes if best_bytes is not None else image_bytes
                    _save_for_download(result_bytes)
                    return result_bytes
                current_img = current_img.resize((new_w, new_h), Image.LANCZOS)

        except Exception:
            # On failure, return original bytes
            try:
                # Attempt to save what we have (original)
                downloads_dir = os.path.join(os.getcwd(), "compressed_downloads")
                os.makedirs(downloads_dir, exist_ok=True)
                filename = f"compressed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_original.jpg"
                file_path = os.path.join(downloads_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
            except Exception:
                pass
            return image_bytes

    def _encode_image_for_model(self, image_bytes: bytes) -> str:
        """Compress then Base64-encode the image for Groq image_url usage."""
        compressed = self._compress_like_whatsapp(image_bytes)
        return base64.b64encode(compressed).decode('utf-8')

    # ============================================================================
    # READING NORMALIZATION & COMPARISON
    # ============================================================================

    def _normalize_reading_str(self, value: Any) -> str:
        """
        Normalize a meter reading string for comparison across models by:
        - Lowercasing
        - Keeping only digits and decimal separators ('.', ',')
        - Removing all decimal separators entirely (ignore decimals)
        - Stripping leading zeros
        - Converting empty result to '0'
        This is ONLY for equality checks between two extractions; it must NOT be
        used for the final value returned to clients.
        """
        if value is None:
            return ""
        try:
            text = str(value).strip().lower()
            # Early exit if explicitly not visible
            if text == "not visible":
                return ""
            allowed_chars = set("0123456789.,")
            filtered = "".join(ch for ch in text if ch in allowed_chars)
            # Remove all decimal separators to ignore decimals for comparison
            digits_only = filtered.replace(".", "").replace(",", "")
            # Strip leading zeros
            no_leading_zeros = digits_only.lstrip('0')
            if no_leading_zeros == "":
                return "0"
            return no_leading_zeros
        except Exception:
            return ""

    def _readings_match(self, a: Any, b: Any) -> bool:
        """
        Compare two readings for semantic equality. This comparison is lenient
        and ignores units, spaces, case, and decimal comma vs dot.
        """
        na = self._normalize_reading_str(a)
        nb = self._normalize_reading_str(b)
        return bool(na) and na == nb

    # ============================================================================
    # MODEL 1: GROQ EXTRACTION
    # ============================================================================

    async def _extract_reading_groq(self, base64_image: str, model: str) -> Any:
        """
        Extract the exact meter register reading using Groq model.
        Returns the raw reading string or None.
        """
        try:
            prompt = (
                "You are an OCR extractor specialized in utility meters (electricity kWh, water m³, gas, etc.).\n\n"
                "Task:\n"
                "- Extract the exact meter register reading as displayed.\n"
                "- Include units if visible (e.g., '37856.3 kWh', '08215 m³'); otherwise return only the number.\n"
                "- Preserve leading zeros and the decimal/comma separator exactly as shown.\n\n"
                "If any digit is unclear, or the reading is not visible, return 'Not visible'.\n\n"
                "Return only the reading value or 'Not visible' - no additional text."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0,
                max_tokens=128,
            )
            
            # Extract reading from response
            reading = chat_completion.choices[0].message.content.strip()
            
            # Clean up the response
            if reading.lower() == "not visible" or reading.lower() == "null" or reading == "":
                return None
            return reading
            
        except Exception as e:
            print(f"Groq extraction error: {e}")
            return None


    # ============================================================================
    # MODEL 2: AWS BEDROCK - Claude
    # ============================================================================

    async def _extract_reading_aws_1(self, base64_image: str) -> Any:
        """
        Extract the exact meter register reading using AWS Bedrock Claude model (Model 1).
        Returns the raw reading string or None.
        """
        try:
            prompt = (
                "You are an OCR extractor specialized in utility meters (electricity kWh, water m³, gas, etc.).\n\n"
                "Task:\n"
                "- Extract the exact meter register reading as displayed.\n"
                "- Include units if visible (e.g., '37856.3 kWh', '08215 m³'); otherwise return only the number.\n"
                "- Preserve leading zeros and the decimal/comma separator exactly as shown.\n\n"
                "If any digit is unclear, or the reading is not visible, return 'Not visible'.\n\n"
                "Return only the reading value or 'Not visible' - no additional text."
            )
            
            # Claude API format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 128,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.aws_model_1_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            result = json.loads(response["body"].read())
            reading = result.get("content", [{}])[0].get("text", "").strip()
            
            # Clean up the response
            if reading.lower() == "not visible" or reading.lower() == "null" or reading == "":
                return None
            return reading
            
        except Exception as e:
            print(f"AWS Bedrock Model 1 (Claude) extraction error: {e}")
            return None

    # ============================================================================
    # MODEL 3: AWS BEDROCK - Nova Pro
    # ============================================================================

    async def _extract_reading_aws_2(self, base64_image: str) -> Any:
        """
        Extract the exact meter register reading using AWS Bedrock Amazon Nova Pro model (Model 2).
        Returns the raw reading string or None.
        """
        try:
            prompt = (
                "You are an OCR extractor that reads meter display values from images.\n\n"
                "Your ONLY job is to return the exact numeric reading as it appears on the meter display.\n\n"
                "Rules:\n"
                "- Output only the exact reading string — no explanations, notes, or additional words.\n"
                "- Include units only if they are clearly visible in the image (e.g., '37856.3 kWh', '08215 m³').\n"
                "- Preserve all digits exactly as shown, including leading zeros, decimals, or commas.\n"
                "- If the reading is unclear or unreadable, return exactly: Not visible\n\n"
                "Final Output Format:\n"
                "Return only one of the following:\n"
                "- The exact meter reading string (e.g., '004567.3', '81245 kWh', '00123 m³')\n"
                "- Or the phrase 'Not visible'\n\n"
                "Do not include any explanation, commentary, or extra words."
            )

            
            # Nova Pro API format
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": base64_image
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.aws_model_2_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            result = json.loads(response["body"].read())
            print(result)
            
            # Nova Pro has different response structure
            if "output" in result and "message" in result["output"]:
                content = result["output"]["message"].get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    reading = content[0].get("text", "").strip()
                else:
                    reading = ""
            else:
                reading = ""
            
            # Clean up the response
            if reading.lower() == "not visible" or reading.lower() == "null" or reading == "":
                return None
            return reading
            
        except Exception as e:
            print(f"AWS Bedrock Model 2 (Nova Pro) extraction error: {e}")
            return None

    # ============================================================================
    # VALIDATION & CONFIDENCE CALCULATION
    # ============================================================================

    async def validate_and_verify_meter_reading_with_confidence_percentage(
        self, 
        image_bytes: bytes, 
        groq_reading: Any = None, 
        *, 
        aws_1_reading: Any = None, 
        aws_2_reading: Any = None, 
        base64_image: str = None
    ) -> Dict[str, Any]:
        """
        Validate and verify the meter reading using three models simultaneously.
        Returns confidence based on agreement:
        - All three match: 93-97% confidence (random)
        - Two match: 75-85% confidence (random)
        - All different: 40-50% confidence (random)
        """
        try:
            # Reuse provided base64 if available; otherwise create it
            if not base64_image:
                base64_image = self._encode_image_for_model(image_bytes)

            # Clean up readings
            groq_reading = groq_reading
            aws_1_reading = aws_1_reading
            aws_2_reading = aws_2_reading
            
            # Handle None values
            if groq_reading is None:
                groq_reading = "Not visible"
            if aws_1_reading is None:
                aws_1_reading = "Not visible"
            if aws_2_reading is None:
                aws_2_reading = "Not visible"
            
            # Convert to strings and clean
            groq_reading = str(groq_reading).strip()
            aws_1_reading = str(aws_1_reading).strip()
            aws_2_reading = str(aws_2_reading).strip()
            
            readings = [groq_reading, aws_1_reading, aws_2_reading]
            
            # Count "Not visible" readings
            not_visible_count = sum(1 for r in readings if r.lower() == "not visible")
            
            # If most models can't detect reading
            if not_visible_count >= 2:
                return {"reading": "Not visible", "confidence": "0%", "reason": "Most models could not detect reading"}
            
            # Check for matches using normalized comparison
            matches = []
            if self._readings_match(groq_reading, aws_1_reading):
                matches.append((groq_reading, aws_1_reading))
            if self._readings_match(groq_reading, aws_2_reading):
                matches.append((groq_reading, aws_2_reading))
            if self._readings_match(aws_1_reading, aws_2_reading):
                matches.append((aws_1_reading, aws_2_reading))
            
            # Determine confidence and final reading
            if len(matches) >= 2:  # All three match (or 2 out of 3 with one "Not visible")
                confidence = random.randint(95, 97)
                final_reading = groq_reading  # Use primary Groq reading
                reason = "All models agree on reading"
            elif len(matches) == 1:  # Two models match
                confidence = random.randint(75, 85)
                # Use the reading from the matching pair
                final_reading = matches[0][0]
                reason = "Two models agree on reading"
            else:  # All different
                confidence = random.randint(40, 50)
                final_reading = groq_reading  # Default to primary Groq reading
                reason = "Models disagree on reading"
            
            return {"reading": final_reading, "confidence": f"{confidence}%", "reason": reason}

        except Exception as e:
            return {"reading": "Not visible", "confidence": "0%", "reason": f"Internal error: {str(e)}"}

    # ============================================================================
    # MAIN PROCESSING METHOD
    # ============================================================================

    async def groq_process_meter_image_from_bytes(
        self, 
        image_bytes: bytes, 
        *, 
        groq_model: str = None
    ) -> Dict[str, Any]:
        """
        Process any type of meter image (electricity, water, gas, etc.) and return the numeric meter reading (e.g., kWh, cubic meters)
        along with a confidence percentage for that reading.
        
        Uses three models simultaneously for validation:
        - AWS Model 1 (Claude) - PRIMARY
        - AWS Model 2 (Nova Pro)
        - Groq (secondary)
        """
        try:
            # Log original image size
            vv = len(image_bytes)
            print("Image bytes length: ")
            print(vv/1024)  # Print in KB
            
            # Compress image and encode to base64
            compressed = self._compress_like_whatsapp(image_bytes)
            base64_image = base64.b64encode(compressed).decode('utf-8')

            # Log compressed image size
            dddd = len(compressed)
            print("After Compressed bytes length: ")
            print(dddd/1024)  # Print in KB
            
            # Run all three models simultaneously (AWS Model 1 is primary)
            groq_model = groq_model or self.groq_model
            
            # Create tasks for parallel execution: AWS 1 (primary), AWS 2, Groq (secondary)
            aws_1_task = self._extract_reading_aws_1(base64_image)
            aws_2_task = self._extract_reading_aws_2(base64_image)
            groq_task = self._extract_reading_groq(base64_image, groq_model)
            
            # Wait for all three to complete
            aws_1_reading, aws_2_reading, groq_reading = await asyncio.gather(
                aws_1_task, aws_2_task, groq_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(aws_1_reading, Exception):
                print(f"AWS Model 1 extraction error: {aws_1_reading}")
                aws_1_reading = None
            if isinstance(aws_2_reading, Exception):
                print(f"AWS Model 2 extraction error: {aws_2_reading}")
                aws_2_reading = None
            if isinstance(groq_reading, Exception):
                print(f"Groq extraction error: {groq_reading}")
                groq_reading = None

            # Log all readings (AWS Model 1 is primary)
            print("AWS Model 1 (Claude) - PRIMARY: ", aws_1_reading)
            print("AWS Model 2 (Nova Pro): ", aws_2_reading)
            print("Groq reading: ", groq_reading)

            # Validate and calculate confidence
            return await self.validate_and_verify_meter_reading_with_confidence_percentage(
                image_bytes,
                groq_reading,
                aws_1_reading=aws_1_reading,
                aws_2_reading=aws_2_reading,
                base64_image=base64_image,
            )
            
        except Exception as e:
            print(e)
            return {"reading": "Not visible", "confidence": "0%"}
   