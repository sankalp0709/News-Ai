import re


class EvidenceEngine:
    """
    Generates explainable evidence
    for extracted entities.
    """

    def generate(self, text: str, entities: dict, classification: dict = None):
        evidence_report = {}

        classification = classification or {}

        matched_keywords = classification.get("matched_keywords", [])

        # Sanitize the incoming text to avoid binary or control characters leaking into
        # the JSON response (often caused by PDFs, binary downloads, or bad decoding).
        # If text is bytes, decode with replacement for invalid sequences.
        if isinstance(text, (bytes, bytearray)):
            try:
                text = text.decode("utf-8", errors="replace")
            except Exception:
                text = str(text)

        # Normalize newlines and remove C0/C1 control characters (except newline)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Replace other non-printable/control characters with a single space
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

        # Collapse multiple whitespace to single space but preserve paragraph breaks
        # First split into paragraphs by newline, then normalize internal whitespace
        paragraphs = []
        for p in text.split("\n"):
            p = re.sub(r"\s+", " ", p).strip()
            if p:
                paragraphs.append(p)

        for entity_group, values in entities.items():
            for entity in values:
                entity_evidence = []
                for paragraph_index, paragraph in enumerate(paragraphs, start=1):
                    # Ensure paragraph is a string
                    if not isinstance(paragraph, str):
                        paragraph = str(paragraph)

                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

                    for sentence in sentences:
                        if not isinstance(sentence, str):
                            sentence = str(sentence)

                        # Lowercase checks should operate on safe strings
                        try:
                            contains_entity = entity.lower() in sentence.lower()
                        except Exception:
                            contains_entity = False

                        if contains_entity:
                            classification_matches = [
                                keyword
                                for keyword in matched_keywords
                                if isinstance(keyword, str) and keyword.lower() in sentence.lower()
                            ]

                            clean_sentence = sentence.strip()

                            entity_evidence.append({
                                "paragraph": paragraph_index,
                                "sentence": clean_sentence,
                                "entity": entity,
                                "classification_keywords": classification_matches,
                                "reason": "Entity found in article",
                                "confidence": 1.0 if classification_matches else 0.8,
                            })

                if entity_evidence:
                    evidence_report[entity] = entity_evidence
        return evidence_report