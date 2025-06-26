import unicodedata
import re
import os

from typing import List
from langchain.schema import Document
import asyncio
from pathlib import Path
from mistralai import Mistral
from mistralai.models import DocumentURLChunk
import json
from config import Config


def clean_text(text: str) -> str:
    """Apply universal text cleaning across all file types"""
    if not text:
        return text
    
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)
    
    # Replace common problematic characters
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        """: '"', """: '"', "'": "'", "'": "'", "′": "'",
        "‒": "-", "–": "-", "—": "-", "―": "-",
        "…": "...", "•": "*", "°": " degrees ",
        "©": "(c)", "®": "(R)", "™": "(TM)"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean control characters while preserving whitespace
    text = "".join(
        char for char in text
        if unicodedata.category(char)[0] != "C" or char in "\n\t "
    )
    
    # Clean up spacing
    text = re.sub(r"[ \t]+", " ", text)  # Consolidate horizontal whitespace
    text = re.sub(r" +\n", "\n", text)  # Remove spaces before newlines
    text = re.sub(r"\n +", "\n", text)  # Remove spaces after newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max two consecutive newlines
    text = re.sub(r"^\s+", "", text)  # Remove leading whitespace
    text = re.sub(r"\s+$", "", text)  # Remove trailing whitespace
    
    # Clean up punctuation spacing
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    text = re.sub(r"(\()\s+", r"\1", text)
    
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u200e\u200f]", "", text)
    
    # Fix hyphenation at line breaks
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    
    return text.strip()

def detect_file_type(file_path: str, filename: str = None) -> str:
    """Detect file type using MIME types and extensions"""
    # Try MIME type detection first
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    
    # Fallback to extension
    ext = os.path.splitext(filename or file_path)[1].lower()
    
    ext_mime_map = {
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }
    
    return ext_mime_map.get(ext, "text/plain")




async def extract_pdf_ocr(file_path: str, filename: str) -> List[Document]:
    """Extract text from PDF files using Mistral OCR - one Document per page"""
    from pathlib import Path
    from mistralai import Mistral
    from mistralai.models import DocumentURLChunk
    import json
    
    def _extract():
        # Initialize client
        client = Mistral(api_key=Config.MISTRAL_OCR_API_KEY)
        
        # Verify PDF file exists
        pdf_file = Path(file_path)
        if not pdf_file.is_file():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Upload PDF file to Mistral's OCR service
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_file.stem,
                "content": pdf_file.read_bytes(),
            },
            purpose="ocr",
        )
        
        # Get URL for the uploaded file
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        # Process PDF with OCR
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=False  # We don't need images for text extraction
        )
        
        # Convert response to dictionary
        response_dict = json.loads(pdf_response.model_dump_json())
        
        # Extract pages data
        pages_data = []
        for page_num, page in enumerate(response_dict["pages"]):
            text = page["markdown"]  # Using markdown format which includes text structure
            if text.strip():  # Only include pages with content
                pages_data.append({
                    "content": text,
                    "page_number": page_num  # 0-indexed
                })
        
        return pages_data
    
    pages_data = await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    documents = []
    for page_data in pages_data:
        cleaned_content = clean_text(page_data["content"])
        if cleaned_content.strip():
            doc = Document(
                page_content=cleaned_content,
                metadata={
                    "source": filename,
                    "filename": filename,
                    "page": page_data["page_number"]
                }
            )
            documents.append(doc)
    
    return documents