import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MarkdownAdapter:
    """Markdown extraction adapter."""
    
    @staticmethod
    def extract(file_content: bytes, filename: str) -> Dict[str, Any]:
        if not file_content:
            raise ValueError("Empty Markdown file")
            
        try:
            # Safe encoding handling
            text = file_content.decode('utf-8', errors='replace')
            
            return {
                "format": "markdown",
                "filename": filename,
                "content": {
                    "text": text.strip(),
                    "metadata": {}
                }
            }
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse Markdown: {str(e)}")
