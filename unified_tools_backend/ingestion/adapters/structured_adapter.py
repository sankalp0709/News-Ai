import csv
import io
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StructuredAdapter:
    """
    Structured data ingestion adapter for JSON and CSV formats.
    Parses structured files safely, preserves raw structured representations,
    and produces normalized text representations for canonical intelligence processing.
    """

    @staticmethod
    def extract_json(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Safely parse JSON content and generate normalized representation.
        """
        if not file_content:
            raise ValueError("Empty JSON file")

        try:
            text = file_content.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid character encoding in JSON: {str(e)}")

        if not text.strip():
            raise ValueError("Empty JSON content")

        try:
            parsed_data = json.loads(text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")

        # Create human-readable and NLP-friendly text representation
        if isinstance(parsed_data, (dict, list)):
            text_repr = json.dumps(parsed_data, indent=2, ensure_ascii=False)
        else:
            text_repr = str(parsed_data)

        if not text_repr.strip():
            raise ValueError("Parsed JSON produced empty text representation")

        item_count = len(parsed_data) if isinstance(parsed_data, (dict, list)) else 1

        return {
            "format": "json",
            "filename": filename,
            "content": {
                "text": text_repr.strip(),
                "structured_data": parsed_data,
                "metadata": {
                    "data_type": type(parsed_data).__name__,
                    "item_count": item_count,
                },
            },
        }

    @staticmethod
    def extract_csv(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Safely parse CSV content, preserve headers and rows, and generate normalized text.
        """
        if not file_content:
            raise ValueError("Empty CSV file")

        try:
            text = file_content.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid character encoding in CSV: {str(e)}")

        if not text.strip():
            raise ValueError("Empty CSV content")

        try:
            reader = csv.reader(io.StringIO(text))
            all_rows = [row for row in reader if any(cell.strip() for cell in row)]
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {str(e)}")

        if not all_rows:
            raise ValueError("CSV contains no valid data rows")

        headers = all_rows[0]
        data_rows = all_rows[1:]

        # Build normalized textual representation for intelligence engine
        text_lines = [f"CSV Table: {filename}"]
        text_lines.append(f"Headers: {', '.join(headers)}")

        if data_rows:
            for idx, row in enumerate(data_rows, start=1):
                row_items = []
                for h_idx, cell in enumerate(row):
                    header_name = headers[h_idx] if h_idx < len(headers) else f"Column_{h_idx + 1}"
                    row_items.append(f"{header_name}: {cell.strip()}")
                text_lines.append(f"Row {idx}: {', '.join(row_items)}")
        else:
            text_lines.append("No additional data rows.")

        text_repr = "\n".join(text_lines)

        return {
            "format": "csv",
            "filename": filename,
            "content": {
                "text": text_repr.strip(),
                "headers": headers,
                "rows": data_rows,
                "row_count": len(data_rows),
                "metadata": {
                    "headers": headers,
                    "row_count": len(data_rows),
                    "column_count": len(headers),
                },
            },
        }

    @classmethod
    def extract(
        cls,
        file_content: bytes,
        filename: str,
        format_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified entry point for structured ingestion.
        Routes to JSON or CSV extractors based on format_type or file extension.
        """
        fmt = (format_type or "").lower()
        if not fmt:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext == "json":
                fmt = "json"
            elif ext == "csv":
                fmt = "csv"

        if fmt == "json":
            return cls.extract_json(file_content, filename)
        elif fmt == "csv":
            return cls.extract_csv(file_content, filename)
        else:
            raise ValueError(f"Unsupported structured format: {format_type or filename}")
