from datetime import datetime, timezone
import hashlib
import re
import time
import uuid

from pydantic import json

from runtime.replay_store import ReplayStore

from analysis.vision_runtime_client import VisionRuntimeClient
from analysis.news_intelligence_service import NewsIntelligenceService

from analysis.svacs_intelligence_mapper import (
    SVACSIntelligenceMapper
)

class VisionIntelligenceService:
    """
    Orchestrates image-based intelligence processing.

    Flow:
    Image
        -> Vision Runtime
        -> OCR Normalization
        -> Samachar Intelligence Engine
        -> Canonical Structured Intelligence

    Vision processing remains owned by the external
    Vision Runtime.
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(self):
        self.vision_client = VisionRuntimeClient()
        self.intelligence_service = NewsIntelligenceService()
        self.svacs_mapper = SVACSIntelligenceMapper()
        
    def process(
        self,
        image_bytes: bytes,
        filename: str,
        execution_context: dict,
        content_type: str = "image/jpeg",
        return_explainable_image: bool = False
    ) -> dict:

        total_start = time.perf_counter()

        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("Image bytes are required")

        execution_id = execution_context["execution_id"]
        trace_id = execution_context["trace_id"]

        processing_times = {}

        input_fingerprint = (
            "sha256:"
            + hashlib.sha256(image_bytes).hexdigest()
        )

        print(f"[Replay] Fingerprint: {input_fingerprint}")

        replay_record = ReplayStore.get(
            input_fingerprint
        )

        if replay_record is not None:
            print("\n========== REPLAY HIT ==========")

            replay_result = replay_record["result"]

            replay_result["replay"] = {
                "status": "HIT",
                "input_fingerprint": input_fingerprint,
                "original_trace_id": replay_record["trace_id"],
            }

            print(replay_result["replay"])

            return replay_result

        print("\n========== REPLAY MISS ==========")

        # ==========================================
        # 1. Vision Runtime Invocation
        # ==========================================

        start = time.perf_counter()

        print("[Vision Runtime] Calling Vision Runtime...")

        vision_result = self.vision_client.analyze_image(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
            return_explainable_image=return_explainable_image
        )

        print("\n========== RAW VISION RUNTIME RESULT ==========")
        print(vision_result)
        print("===============================================\n")

        processing_times["vision_runtime"] = round(
            time.perf_counter() - start,
            3
        )

        # ==========================================
        # 2. OCR Normalization
        # ==========================================

        start = time.perf_counter()

        raw_ocr_results = vision_result.get(
            "ocr_results",
            []
        )

        normalized_ocr_results = (
            self._normalize_ocr_results(
                raw_ocr_results
            )
        )

        normalized_ocr_text = " ".join(
            item["text"]
            for item in normalized_ocr_results
        )

        processing_times["ocr_normalization"] = round(
            time.perf_counter() - start,
            3
        )

        # ==========================================
        # 3. Samachar Intelligence Processing
        # ==========================================

        intelligence = None

        if normalized_ocr_text:

            start = time.perf_counter()

            intelligence_input = {
                "title": "",
                "content": normalized_ocr_text,
                "publication_date": ""
            }

            intelligence = (
                self.intelligence_service.process(
                    intelligence_input
                )
            )

            processing_times[
                "intelligence_processing"
            ] = round(
                time.perf_counter() - start,
                3
            )

        else:

            processing_times[
                "intelligence_processing"
            ] = 0.0

        # ==========================================
        # 4. Canonical Mapping
        # ==========================================

        start = time.perf_counter()

        canonical_response = {
            "schema_version": self.SCHEMA_VERSION,

            "execution_id": execution_id,
            "trace_id": trace_id,

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "source": {
                "input_type": "image",
                "source_system": "samachar",
                "filename": filename
            },

            "provenance": {
                "origin": "operator_image",

                "processed_by": [
                    "samachar",
                    "vision_runtime"
                ],

                "vision_runtime_invoked": True,

                "vision_replay_id": (
                    vision_result.get(
                        "replay_id"
                    )
                ),
                "input_fingerprint": input_fingerprint,
                "normalization": {
                    "ocr_results_received": len(raw_ocr_results),
                    "ocr_results_normalized": len(normalized_ocr_results),
                },
            },

            "vision_intelligence": {
                "replay_id": vision_result.get(
                    "replay_id"
                ),

                "detections": vision_result.get(
                    "detections",
                    []
                ),

                # Preserve Vijay's raw OCR response
                "ocr_results": raw_ocr_results,

                # Explicit Samachar normalization
                "normalized_ocr_results": (
                    normalized_ocr_results
                ),

                "explainable_image_base64": (
                    vision_result.get(
                        "explainable_image_base64"
                    )
                )
            },

            "intelligence": intelligence,

            "processing_trace": {
                "status": "SUCCESS",
                "execution_id": execution_id,
                "trace_id": trace_id,
                "vision_replay_id": vision_result.get("replay_id"),

                "steps": [
                    {
                        "name": "Image Ingestion",
                        "status": "SUCCESS"
                    },
                    {
                        "name": "Vision Runtime",
                        "status": "SUCCESS"
                    },
                    {
                        "name": "OCR Normalization",
                        "status": "SUCCESS"
                    },
                    {
                        "name": "Samachar Intelligence",
                        "status": "SUCCESS"
                    },
                    {
                        "name": "Canonical Mapping",
                        "status": "SUCCESS"
                    }
                ],

                "processing_time": processing_times
            },

            "downstream": {
                "target_system": "svacs",
                "ready_for_processing": True
            },

            "replay": {
                    "status": "MISS",
                    "input_fingerprint": (
                        input_fingerprint
                    ),
                    "original_trace_id": trace_id,
                },

                "errors": []
            }

        processing_times["canonical_mapping"] = round(
                time.perf_counter() - start,
                3
            )

        processing_times["total"] = round(
                time.perf_counter() - total_start,
                3
            )

        print("\nSaving result into ReplayStore...")
        print(canonical_response["replay"])

        ReplayStore.save(
                input_fingerprint=input_fingerprint,
                trace_id=trace_id,
                input_type="image",
                schema_version=self.SCHEMA_VERSION,
                result=canonical_response,
            )

        print(canonical_response["processing_trace"])

        return canonical_response

    def _normalize_ocr_results(
        self,
        ocr_results: list
    ) -> list:
        """
        Creates normalized OCR input for Samachar intelligence.

        Raw Vision Runtime OCR results remain unchanged
        in the canonical response.

        Current normalization:
        - Minimum confidence threshold
        - Remove surrounding punctuation
        - Remove exact duplicate OCR text

        This method does not perform OCR.
        """

        normalized_results = []

        seen_text = set()

        minimum_confidence = 0.60

        for item in ocr_results:

            text = item.get(
                "text",
                ""
            ).strip()

            confidence = item.get(
                "confidence",
                0
            )

            if not text:
                continue

            if confidence < minimum_confidence:
                continue

            text = re.sub(
                r'^[\'"“”]+|[\'"“”]+$',
                "",
                text
            ).strip()

            if not text:
                continue

            normalized_key = text.lower()

            if normalized_key in seen_text:
                continue

            seen_text.add(
                normalized_key
            )

            normalized_results.append({
                "text": text,
                "confidence": confidence,
                "source": "vision_runtime_ocr"
            })

        return normalized_results
