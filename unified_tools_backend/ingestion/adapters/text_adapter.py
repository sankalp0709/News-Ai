import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TextAdapter:
    """TXT extraction adapter."""
    
    @staticmethod
    def extract(file_content: bytes, filename: str) -> Dict[str, Any]:
        if not file_content:
            raise ValueError("Empty TXT file")
            
        try:
            # Safe encoding handling
            text = file_content.decode('utf-8', errors='replace')
            
            return {
                "format": "txt",
                "filename": filename,
                "content": {
                    "text": text.strip(),
                    "metadata": {}
                }
            }
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse TXT: {str(e)}")
