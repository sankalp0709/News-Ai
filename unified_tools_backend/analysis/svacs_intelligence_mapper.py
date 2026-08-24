import re

class SVACSIntelligenceMapper:
    """
    Maps Samachar canonical intelligence into the
    SVACS structured intelligence contract.

    This mapper performs schema translation only.

    It does not perform:
    - vessel detection
    - maritime reasoning
    - image classification
    - dimension estimation
    - sensor fusion
    """

    SCHEMA_VERSION = "1.0.0"

    ALLOWED_VESSEL_CLASSES = {
        "cargo",
        "tanker",
        "patrol",
        "fishing",
        "submarine",
        "unknown",
    }

    VESSEL_CLASS_MAPPING = {
        # Cargo
        "cargo": "cargo",
        "cargo ship": "cargo",
        "cargo vessel": "cargo",
        "container ship": "cargo",
        "container vessel": "cargo",

        # Passenger ferry
        "passenger ferry": "passenger ferry",

        # Tankers
        "tanker": "tanker",
        "oil tanker": "tanker",
        "chemical tanker": "tanker",
        "lng tanker": "tanker",

        # Patrol
        "patrol": "patrol",
        "patrol boat": "patrol",
        "patrol vessel": "patrol",

        # Fishing
        "fishing": "fishing",
        "fishing boat": "fishing",
        "fishing vessel": "fishing",

        # Submarine
        "submarine": "submarine",

        # Default / Generic
        "ship": "unknown",
    }

    def map(self, canonical_intelligence: dict) -> dict:
        """
        Convert Samachar canonical intelligence
        into the SVACS structured intelligence schema.
        """

        if not canonical_intelligence:
            raise ValueError(
                "Canonical intelligence cannot be empty"
            )

        trace_id = canonical_intelligence.get(
            "trace_id"
        )

        if not trace_id:
            raise ValueError(
                "Samachar trace_id is required"
            )

        source = (
            canonical_intelligence.get("source")
            or {}
        )

        vision_intelligence = (
            canonical_intelligence.get(
                "vision_intelligence"
            )
            or {}
        )

        samachar_intelligence = (
            canonical_intelligence.get(
                "intelligence"
            )
            or {}
        )

        detections = (
            vision_intelligence.get(
                "detections"
            )
            or []
        )

        ocr_results = (
            vision_intelligence.get(
                "ocr_results"
            )
            or []
        )

        normalized_ocr_results = (
            vision_intelligence.get(
                "normalized_ocr_results"
            )
            or []
        )

        primary_detection = (
            detections[0]
            if (
                detections
                and isinstance(
                    detections[0],
                    dict
                )
            )
            else {}
        )

        vision_label = primary_detection.get(
            "label",
            ""
        )

        vision_confidence = primary_detection.get(
            "confidence",
            0.0
        )

        vessel_class = self._map_vessel_class(
            vision_label
        )

        maritime_identifier = (
            self._extract_maritime_identifier(
                ocr_results
            )
        )

        svacs_ocr_results = (
            self._build_ocr_results(
                ocr_results=ocr_results,
                normalized_ocr_results=(
                    normalized_ocr_results
                ),
            )
        )

        confidence_report = (
            samachar_intelligence.get(
                "confidence"
            )
            or {}
        )

        if not isinstance(
            confidence_report,
            dict
        ):
            confidence_report = {}

        confidence_score = (
            confidence_report.get(
                "score"
            )
        )

        if confidence_score is None:
            confidence_score = 0.0

        elif isinstance(confidence_score,(int, float)):
            if (confidence_score > 1.0
                and confidence_score <= 100.0
            ):
                confidence_score = (confidence_score / 100.0)

            confidence_score = max(
                0.0,
                min(float(confidence_score),1.0)
            )

        else:
            confidence_score = 0.0

        return {
            "trace_id": trace_id,
            "source_type": source.get(
                "input_type",
                "unknown"
            ),
            "vessel_class": vessel_class,
            "confidence_score": confidence_score,
            "vision_confidence": vision_confidence,
            "ocr_results": svacs_ocr_results,
            "visual_features": [],
            "dimensions_estimate": {
                "length_m": None,
                "beam_m": None,
            },
            "ais_data": {
                "mmsi": maritime_identifier,
                "speed_knots": None,
            },
            "timestamp_utc": (
                canonical_intelligence.get(
                    "timestamp"
                )
            ),
        }

    def _map_vessel_class(self,vision_label: str) -> str:
        """
        Maps Vision Runtime labels to the
        approved SVACS vessel taxonomy.
        """

        if not vision_label:
            return "unknown"

        normalized_label = (
            vision_label
            .strip()
            .lower()
        )

        vessel_class = (
            self.VESSEL_CLASS_MAPPING.get(
                normalized_label,
                "unknown"
            )
        )

        if (
            vessel_class
            not in self.ALLOWED_VESSEL_CLASSES
        ):
            return "unknown"

        return vessel_class

    def _build_ocr_results(
        self,
        ocr_results: list,
        normalized_ocr_results: list,
    ) -> list:
        """
        Preserve Vision Runtime OCR intelligence for
        downstream SVACS registry enrichment.

        Normalized OCR text is preferred when available.

        Original Vision Runtime confidence values are
        preserved without reinterpretation.
        """

        if not isinstance(
            ocr_results,
            list
        ):
            return []

        normalized_lookup = {}

        if isinstance(
            normalized_ocr_results,
            list
        ):
            for result in normalized_ocr_results:

                if not isinstance(
                    result,
                    dict
                ):
                    continue

                text = str(
                    result.get("text")
                    or ""
                ).strip()

                if not text:
                    continue

                normalized_lookup[
                    text.lower()
                ] = result

        mapped_results = []

        for result in ocr_results:

            if not isinstance(
                result,
                dict
            ):
                continue

            raw_text = str(
                result.get("text")
                or ""
            ).strip()

            if not raw_text:
                continue

            clean_text = raw_text.strip(
                "\"' "
            )

            normalized_match = (
                normalized_lookup.get(
                    clean_text.lower()
                )
            )

            if normalized_match:
                output_text = (
                    normalized_match.get(
                        "text"
                    )
                    or clean_text
                )
            else:
                output_text = clean_text

            mapped_results.append(
                {
                    "text": output_text,
                    "confidence": result.get(
                        "confidence",
                        0.0
                    ),
                }
            )

        return mapped_results

    def _extract_maritime_identifier(
        self,
        ocr_results: list
    ):
        """
        Extract an IMO identifier from Vision OCR.

        The value is mapped to ais_data.mmsi only
        to preserve the current SVACS v1 contract.

        This compatibility mapping should be reviewed
        if SVACS separates IMO and MMSI identifiers.
        """

        for result in ocr_results:

            if not isinstance(
                result,
                dict
            ):
                continue

            text = str(
                result.get(
                    "text"
                )
                or ""
            )

            imo_match = re.search(
                r"\bIMO[\s:\-]*([0-9]{7})\b",
                text,
                re.IGNORECASE,
            )

            if imo_match:
                return imo_match.group(1)

        return None