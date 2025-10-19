import base64
import io
import json
from typing import Dict, Any
import os
from dotenv import load_dotenv
import requests
from PIL import Image
import pillow_heif
from datetime import datetime

load_dotenv()


class MeterService:
    def __init__(self):
        # Ollama settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = "llava:7b"
        # Default model choices; can be overridden per-call (all using llava now)
        self.default_extract_model = "llava:7b"
        self.default_validate_model = "llava:7b"
        self.default_scout_model = "llava:7b"

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

    async def _scout_extract_reading(self, base64_image: str, model: str = None) -> Any:
        """
        Extract reading using the Scout-style model (or any provided model) for
        double confirmation.
        """
        return await self._extract_reading(base64_image, model or self.default_scout_model)

    def _encode_image_for_model(self, image_bytes: bytes) -> str:
        """
        Compress then Base64-encode the image for Ollama vision model usage.
        """
        compressed = self._compress_like_whatsapp(image_bytes)
        return base64.b64encode(compressed).decode('utf-8')

    def _ollama_vision_chat(self, *, model: str, prompt: str, base64_image: str, temperature: float = 0) -> str:
        """
        Call Ollama API with vision model and return the response text.
        """
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            raise

    async def _extract_reading(self, base64_image: str, model: str) -> Any:
        """
        Extract the exact meter register reading using the specified model.
        Returns the raw reading string or None.
        """
        try:
            prompt = (
                "You are an OCR extractor specialized in utility meters (electricity kWh, water m³, gas, etc.).\n\n"
                "Task:\n"
                "- Extract the exact meter register reading as displayed.\n"
                "- Include units if visible (e.g., '37856.3 kWh', '08215 m³'); otherwise return only the number.\n"
                "- Preserve leading zeros and the decimal/comma separator exactly as shown.\n\n"
                "If any digit is unclear, or the reading is not visible, respond with 'null'.\n\n"
                "IMPORTANT: Respond with ONLY a JSON object in this exact format: {\"reading\": \"your_reading_here\"} or {\"reading\": null}\n"
                "Do not include any other text or explanation."
            )
            
            response_text = self._ollama_vision_chat(
                model=model,
                prompt=prompt,
                base64_image=base64_image,
                temperature=0,
            )
            
            # Try to parse JSON from response
            # Sometimes models add extra text, so we'll try to extract JSON
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    data = json.loads(json_str)
                    return data.get("reading")
                else:
                    # Try parsing the whole response as JSON
                    data = json.loads(response_text)
                    return data.get("reading")
            except:
                # If JSON parsing fails, try to extract the reading directly
                if "null" in response_text.lower() or "not visible" in response_text.lower():
                    return None
                # Return the cleaned response as the reading
                return response_text.strip()
        except Exception as e:
            print(f"Error extracting reading: {e}")
            return None

    async def _validate_via_model(self, base64_image: str, initial_reading: Any, model: str) -> Dict[str, Any]:
        """
        Validate the reading using the specified model and return a dict with
        final_reading, confidence, and optional reason.
        """
        validation_prompt = (
            "You are validating a utility meter reading from an image.\n\n"
            "Instructions:\n"
            "1) If the main register reading is not fully legible or any digit is uncertain → final_reading = 'Not visible' and provide a brief reason.\n"
            "2) Otherwise, return the exact reading string as displayed (keep leading zeros, punctuation, and units if visible).\n"
            "3) Ignore serial numbers, dates, CT ratios, tariffs, diagnostic values, and anything outside the main register.\n\n"
            "Constraints:\n"
            "- confidence is a percentage string like '83%'. Maximum allowed is '97%'.\n"
            "- If final_reading == 'Not visible', reason is required.\n\n"
        )
        if initial_reading:
            validation_prompt += (
                f"\n\nPrevious reading extracted: {initial_reading}. "
                "Please verify correctness and provide confidence (percentage string, max 97%). "
                "If 'Not visible', include a brief reason.\n\n"
            )
        
        validation_prompt += (
            "IMPORTANT: Respond with ONLY a JSON object in this exact format:\n"
            "{\"final_reading\": \"...\", \"confidence\": \"...%\", \"reason\": \"...\"}\n"
            "Do not include any other text or explanation."
        )

        try:
            response_text = self._ollama_vision_chat(
                model=model,
                prompt=validation_prompt,
                base64_image=base64_image,
                temperature=0,
            )
            
            # Try to parse JSON from response
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    data = json.loads(json_str)
                else:
                    data = json.loads(response_text)
                
                # Ensure required fields exist
                if "final_reading" not in data:
                    data["final_reading"] = "Not visible"
                if "confidence" not in data:
                    data["confidence"] = "0%"
                    
                return data
            except:
                # If JSON parsing fails completely, return default
                return {"final_reading": "Not visible", "confidence": "0%", "reason": "Failed to parse validation response"}
                
        except Exception as e:
            print(f"Validation error: {e}")
            return {"final_reading": "Not visible", "confidence": "0%", "reason": "Validation error"}

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

    async def process_meter_image_from_bytes(self, image_bytes: bytes, *, extract_model: str = None, validate_model: str = None, scout_model: str = None) -> Dict[str, Any]:
        """
        Process any type of meter image (electricity, water, gas, etc.) and return the numeric meter reading (e.g., kWh, cubic meters)
        along with a confidence percentage for that reading.
        """
        try:
            vv = len(image_bytes)
            print("Image bytes length: ")
            # vv in kb
            print(vv/1024) #this will print the image bytes length in kb
            # Compress like WhatsApp, then encode to base64 (and log size)
            compressed = self._compress_like_whatsapp(image_bytes)
            base64_image = base64.b64encode(compressed).decode('utf-8')

            dddd = len(compressed)
            print("After Compressed bytes length: ")
            print(dddd/1024) #this will print the compressed bytes length in kb
            # Primary extraction using configurable model
            primary_model = extract_model or self.default_extract_model
            initial_reading = await self._extract_reading(base64_image, primary_model)

            return await self.validate_and_verify_meter_reading_with_confidence_percentage(
                image_bytes,
                initial_reading,
                validate_model=validate_model,
                scout_model=scout_model,
                base64_image=base64_image,
            )
        except Exception as e:
            print(e)
            return {"reading": "Not visible", "confidence": "0%"}

    async def validate_and_verify_meter_reading_with_confidence_percentage(self, image_bytes: bytes, initial_reading: Any = None, *, validate_model: str = None, scout_model: str = None, base64_image: str = None) -> Dict[str, Any]:
        """
        Validate and verify the meter reading from the image, and return the verified reading, confidence percentage (as a string, e.g. '98%'), 
        and if not visible, also include the reason.
        NOTE: The max confidence returned will be 97%.
        """
        try:
            # Reuse provided base64 if available; otherwise create it
            if not base64_image:
                base64_image = self._encode_image_for_model(image_bytes)

            # Validation via configurable model
            validator_model = validate_model or self.default_validate_model
            data = await self._validate_via_model(base64_image, initial_reading, validator_model)
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

            # Second-pass double confirmation using Scout (or provided) model
            try:
                # Only attempt double-confirm when we have a candidate reading
                candidate_reading = result.get("reading")
                if candidate_reading and candidate_reading.strip().lower() != "not visible":
                    scout_reading = await self._scout_extract_reading(base64_image, model=(scout_model or self.default_scout_model))
                    if scout_reading is not None:
                        # Adjust confidence based on agreement
                        current_conf_str = result.get("confidence", "0%")
                        try:
                            current_conf_val = int(current_conf_str.strip().strip('%')) if current_conf_str.endswith('%') else 0
                        except Exception:
                            current_conf_val = 0

                        if self._readings_match(candidate_reading, scout_reading):
                            boosted = min(97, current_conf_val + 5)
                            result["confidence"] = f"{boosted}%"
                        else:
                            reduced = max(0, current_conf_val - 20)
                            result["confidence"] = f"{reduced}%"
                else:
                    # If initial is not visible, still try scout; if scout yields a reading, we keep Not visible
                    _ = await self._scout_extract_reading(base64_image, model=(scout_model or self.default_scout_model))
            except Exception:
                # Do not fail the request if the second model errors
                pass

            return result

        except Exception as e:
            return {"reading": "Not visible", "confidence": "0%", "reason": "Internal error or exception"}

   