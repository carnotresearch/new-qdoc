"""
Simplified text extraction module for various document formats.

This module provides unified text extraction with automatic file type detection
and consistent output formatting across all supported file types.
"""

import asyncio
import logging
import mimetypes
import os
import re
import tempfile
import unicodedata
from typing import Dict, List, Tuple, Any

# Third-party imports
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain.schema import Document
from openpyxl import load_workbook
from pptx import Presentation
from werkzeug.datastructures import FileStorage

# Configure logging
logger = logging.getLogger(__name__)

def clean_filename(filename: str) -> str:
    """Clean filename by removing temporary path prefixes"""
    if not filename:
        return filename
    
    # Handle both Windows and Unix paths
    if '\\' in filename:
        filename = filename.split('\\')[-1]
    if '/' in filename:
        filename = filename.split('/')[-1]
    
    return filename

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

async def extract_txt(file_path: str) -> str:
    """Extract text from plain text files"""
    def _read_file():
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    content = await asyncio.get_event_loop().run_in_executor(None, _read_file)
    return clean_text(content)

async def extract_pdf(file_path: str) -> str:
    """Extract text from PDF files"""
    def _extract():
        doc = fitz.open(file_path)
        try:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        finally:
            doc.close()
    
    raw_text = await asyncio.get_event_loop().run_in_executor(None, _extract)
    return clean_text(raw_text)

async def extract_docx(file_path: str) -> str:
    """Extract content from DOCX files"""
    def _extract():
        doc = DocxDocument(file_path)
        content = []
        
        for paragraph in doc.paragraphs:
            if not paragraph.text.strip():
                continue
            
            style = paragraph.style.name if paragraph.style else "Normal"
            text = paragraph.text.strip()
            
            # Handle headings
            if "Heading" in style:
                level = style[-1] if style[-1].isdigit() else "1"
                heading_marks = "#" * int(level)
                content.append(f"\n{heading_marks} {text}\n")
            else:
                # Handle text formatting
                formatted_text = []
                for run in paragraph.runs:
                    if run.bold:
                        formatted_text.append(f"**{run.text}**")
                    elif run.italic:
                        formatted_text.append(f"*{run.text}*")
                    else:
                        formatted_text.append(run.text)
                content.append(''.join(formatted_text))
        
        return "\n\n".join(content)
    
    raw_content = await asyncio.get_event_loop().run_in_executor(None, _extract)
    return clean_text(raw_content)

async def extract_pptx(file_path: str) -> str:
    """Extract content from PPTX files"""
    def _extract():
        prs = Presentation(file_path)
        content = []
        
        for slide_number, slide in enumerate(prs.slides, 1):
            content.append(f"\n# Slide {slide_number}\n")
            
            # Extract title
            if slide.shapes.title and slide.shapes.title.text.strip():
                content.append(f"## {slide.shapes.title.text.strip()}\n")
            
            # Extract text from all shapes
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    if shape != slide.shapes.title:
                        content.append(shape.text.strip())
        
        return "\n\n".join(content)
    
    raw_content = await asyncio.get_event_loop().run_in_executor(None, _extract)
    return clean_text(raw_content)

async def extract_xlsx(file_path: str) -> str:
    """Extract content from XLSX files"""
    def _extract():
        wb = load_workbook(file_path, data_only=True)
        content = []
        
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            content.append(f"\n# Sheet: {sheet_name}\n")
            
            max_row = min(ws.max_row or 1, 1000)
            max_col = min(ws.max_column or 1, 50)
            
            if max_row <= 1:
                content.append("(Empty sheet)")
                continue
            
            # Create table
            headers = []
            for col in range(1, max_col + 1):
                cell_value = ws.cell(row=1, column=col).value
                headers.append(str(cell_value) if cell_value is not None else "")
            
            content.append("| " + " | ".join(headers) + " |")
            content.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            # Add data rows
            for row in range(2, max_row + 1):
                row_data = []
                for col in range(1, max_col + 1):
                    cell_value = ws.cell(row=row, column=col).value
                    row_data.append(str(cell_value) if cell_value is not None else "")
                content.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(content)
    
    raw_content = await asyncio.get_event_loop().run_in_executor(None, _extract)
    return clean_text(raw_content)

async def extract_content_from_file(file_path: str, filename: str = None) -> Dict[str, Any]:
    """
    Extract content from any supported file type
    
    Args:
        file_path: Path to the file
        filename: Original filename
        
    Returns:
        Dictionary with extracted content and metadata
    """
    clean_name = clean_filename(filename or os.path.basename(file_path))
    
    # Validate file
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {
            "content": "",
            "metadata": {"filename": clean_name, "error": "File not found or empty"},
            "success": False
        }
    
    # Detect file type
    file_type = detect_file_type(file_path, filename)
    
    try:
        # Extract based on file type
        if file_type == "text/plain":
            content = await extract_txt(file_path)
        elif file_type == "application/pdf":
            content = await extract_pdf(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            content = await extract_docx(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            content = await extract_pptx(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            content = await extract_xlsx(file_path)
        else:
            # Fallback to text
            content = await extract_txt(file_path)
        
        if not content.strip():
            return {
                "content": "",
                "metadata": {"filename": clean_name, "error": "No content extracted"},
                "success": False
            }
        
        return {
            "content": content,
            "metadata": {
                "filename": clean_name,
                "file_type": file_type,
                "content_length": len(content)
            },
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error extracting content from {clean_name}: {e}")
        return {
            "content": "",
            "metadata": {"filename": clean_name, "error": str(e)},
            "success": False
        }

def get_text_from_files(files: List[FileStorage]) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Extract text from multiple files (main interface for document_service.py)
    
    Args:
        files: List of file objects
        
    Returns:
        Tuple of (list of Document objects, list of file_info dictionaries)
    """
    async def _async_extract():
        all_documents = []
        file_infos = []
        
        for file in files:
            temp_path = None
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    file.save(temp_file.name)
                    temp_path = temp_file.name
                
                # Extract content
                result = await extract_content_from_file(temp_path, file.filename)
                
                if result["success"] and result["content"]:
                    logger.info(f"Successfully extracted content from {file.filename}")
                    # Create Document object
                    doc = Document(
                        page_content=result["content"],
                        metadata={
                            "source": clean_filename(file.filename),
                            "filename": clean_filename(file.filename),
                            **result["metadata"]
                        }
                    )
                    all_documents.append(doc)
                
                # File info for tracking
                file_info = {
                    "filename": file.filename,
                    "success": result["success"],
                    "content_length": result["metadata"].get("content_length", 0)
                }
                
                if not result["success"]:
                    file_info["error"] = result["metadata"].get("error", "Unknown error")
                
                file_infos.append(file_info)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                file_infos.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
            finally:
                # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return all_documents, file_infos
    
    # Run async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_async_extract())
    finally:
        loop.close()

# Legacy functions for backward compatibility
def get_text_from_pdf(pdf_file: FileStorage) -> List[Document]:
    """Legacy function for PDF extraction"""
    docs, _ = get_text_from_files([pdf_file])
    return docs

def get_text_from_doc(doc_file: FileStorage) -> List[Document]:
    """Legacy function for DOCX extraction"""
    docs, _ = get_text_from_files([doc_file])
    return docs

def get_text_from_txt(txt_file: FileStorage) -> List[Document]:
    """Legacy function for TXT extraction"""
    docs, _ = get_text_from_files([txt_file])
    return docs

def process_single_file(file: FileStorage) -> Tuple[List[Document], Dict[str, Any]]:
    """Legacy function for single file processing"""
    docs, file_infos = get_text_from_files([file])
    return docs, file_infos[0] if file_infos else {"success": False, "error": "No file info"}

# Standalone usage support
async def extract_from_path(file_path: str) -> Dict[str, Any]:
    """Extract content from a file path (for standalone usage)"""
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    
    return await extract_content_from_file(file_path)

def main():
    """Main function for standalone usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract text from documents")
    parser.add_argument("--file", "-f", help="Path to file")
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.file:
        parser.print_help()
        sys.exit(1)
    
    async def _extract():
        result = await extract_from_path(args.file)
        
        if result["success"]:
            content = result["content"]
            print(f"Successfully extracted text from: {result['metadata']['filename']}")
            print(f"Content length: {len(content)} characters")
            print("-" * 50)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Content saved to: {args.output}")
            else:
                print("Extracted content:")
                print(content[:1000] + "..." if len(content) > 1000 else content)
        else:
            print(f"Error: {result['error']}")
            sys.exit(1)
    
    asyncio.run(_extract())

if __name__ == "__main__":
    main()