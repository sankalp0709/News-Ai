import io
import logging
from typing import Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class DocumentPdfAdapter:
    """PDF extraction adapter."""
    
    @staticmethod
    def extract(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text, page count, and metadata from PDF.
        """
        if not file_content:
            raise ValueError("Empty PDF file")
            
        try:
            reader = PdfReader(io.BytesIO(file_content))
            
            page_count = len(reader.pages)
            if page_count == 0:
                raise ValueError("PDF has no pages")
                
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                    
            metadata = {}
            if reader.metadata:
                for k, v in reader.metadata.items():
                    if v is not None:
                        metadata[k.strip('/')] = str(v)
                    
            return {
                "format": "pdf",
                "filename": filename,
                "content": {
                    "text": text.strip(),
                    "page_count": page_count,
                    "metadata": metadata
                }
            }
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
