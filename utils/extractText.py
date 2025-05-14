"""
Text extraction module for various document formats.

This module provides functions for:
- Extracting text from PDF files
- Extracting text from DOCX files
- Extracting text from TXT files
- Processing images in documents using OCR
"""

# Standard library imports
import base64
import logging
import os

# Third-party imports
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_community.document_loaders.text import TextLoader
import openai
import pdfplumber
import pypdfium2 as pdfium
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def clean_filename(filename):
    """
    Clean a filename by removing temporary path prefixes and standardizing format.
    
    Args:
        filename (str): The filename which may contain 'temp/' prefix
        
    Returns:
        str: Cleaned filename without 'temp/' prefix
    """
    if not filename:
        return filename
        
    # Handle Windows-style paths
    if '\\' in filename:
        filename = filename.split('\\')[-1]
    
    # Handle Unix-style paths
    if '/' in filename:
        filename = filename.split('/')[-1]
    
    return filename


def encode_image(image_path):
    """
    Encode an image file as base64.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64-encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_caption(image_path):
    """
    Get a caption for an image using OpenAI's vision model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Caption describing the text in the image
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4.5-preview",
        messages=[
            {"role": "system", "content": "You are an AI that describes images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Can you describe the text in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
    )
    
    logging.info("------------------------------------")
    logging.info(f"Image desc response: {response}")
    logging.info("------------------------------------")
    
    return response.choices[0].message.content


def get_text_from_doc(doc_file):
    """
    Extract text from a DOCX file.
    
    Args:
        doc_file (werkzeug.datastructures.FileStorage): DOCX file object
        
    Returns:
        list: List of Document objects containing extracted text
    """
    # Save file temporarily to load using langchain tool
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(doc_file.filename))
    
    try:
        doc_file.save(temp_file_path)
        loader = Docx2txtLoader(temp_file_path)
        data = loader.load()
        
        # Clean filename
        clean_file_name = clean_filename(doc_file.filename)
        
        # Add clean filename to metadata
        for doc in data:
            # Always clean the source in metadata
            if 'source' in doc.metadata:
                doc.metadata['source'] = clean_filename(doc.metadata['source'])
            else:
                doc.metadata['source'] = clean_file_name
                
            # Add clean filename to metadata
            doc.metadata['filename'] = clean_file_name
            
        return data
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def get_text_from_txt(txt_file):
    """
    Extract text from a TXT file.
    
    Args:
        txt_file (werkzeug.datastructures.FileStorage): TXT file object
        
    Returns:
        list: List of Document objects containing extracted text
    """
    # Save file temporarily to load using langchain tool
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(txt_file.filename))
    
    try:
        txt_file.save(temp_file_path)
        loader = TextLoader(temp_file_path)
        data = loader.load()
        
        # Clean filename
        clean_file_name = clean_filename(txt_file.filename)
        
        # Add clean filename to metadata
        for doc in data:
            # Always clean the source in metadata
            if 'source' in doc.metadata:
                doc.metadata['source'] = clean_filename(doc.metadata['source'])
            else:
                doc.metadata['source'] = clean_file_name
                
            # Add clean filename to metadata
            doc.metadata['filename'] = clean_file_name
            
        return data
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def get_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file (werkzeug.datastructures.FileStorage): PDF file object
        
    Returns:
        list: List of Document objects containing extracted text
    """
    # Save file temporarily to load using langchain tool
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
    pdf_file.save(temp_file_path)

    # Initialize data list
    data = []
    
    # Option to convert pages to images for OCR scanning (currently disabled)
    '''with pdfplumber.open(temp_file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = scan_doc(temp_file_path)
            if text:
                document = Document(
                    metadata={'source': temp_file_path},
                    page_content=text
                )
                data.append(document)'''
    
    try:
        # Use PyMuPDF for text extraction
        loader = PyMuPDFLoader(temp_file_path)
        data = loader.load()
        
        # Clean filename and add to metadata if not present
        clean_file_name = clean_filename(pdf_file.filename)
        
        for doc in data:
            # Always clean the source in metadata
            if 'source' in doc.metadata:
                doc.metadata['source'] = clean_filename(doc.metadata['source'])
            else:
                doc.metadata['source'] = clean_file_name
                
            # Add clean filename to metadata
            doc.metadata['filename'] = clean_file_name
        
        logging.info(f'Extracted data len: {len(data)}')
        return data
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_single_file(file):
    """
    Process a single file to extract text.
    
    Args:
        file (werkzeug.datastructures.FileStorage): File object
        
    Returns:
        tuple: (list of Document objects, file_info dictionary)
    """
    logging.info(f'Processing single file: {file.filename}')
    
    file_info = {
        'filename': file.filename,
        'file_type': file.filename.split('.')[-1].lower(),
        'success': False,
        'page_count': 0
    }
    
    # Extract text based on file type
    if file.filename.endswith(".pdf"):
        data = get_text_from_pdf(file)
    elif file.filename.endswith((".doc", ".docx")):
        data = get_text_from_doc(file)
    elif file.filename.endswith(".txt"):
        data = get_text_from_txt(file)
    else:
        logging.info(f"Unsupported file type: {file.filename}")
        file_info['error'] = "Unsupported file type"
        return [], file_info
    
    # Update file info
    if data:
        file_info['success'] = True
        file_info['page_count'] = len(data)
    
    return data, file_info


def get_text_from_files(files):
    """
    Extract text from multiple files of various formats.
    
    Args:
        files (list): List of file objects
        
    Returns:
        tuple: (list of Document objects, list of file_info dictionaries)
    """
    all_text = []  # All extracted documents
    file_infos = []  # Information about processed files
    
    for file in files:
        docs, file_info = process_single_file(file)
        all_text.extend(docs)
        file_infos.append(file_info)
    
    return all_text, file_infos


def scan_doc(file):
    """
    Extract text from a document by converting pages to images and using OCR.
    
    Args:
        file (str): Path to the document file
        
    Returns:
        str: Extracted text
    """
    pdf = pdfium.PdfDocument(file)

    # Text file to store captions
    with open("temp/captions.txt", "w", encoding="utf-8") as caption_file:
        # Loop over pages and render each one as an image
        for i in range(len(pdf)):
            page = pdf[i]
            image = page.render(scale=4).to_pil()

            # Save image temporarily
            image_path = f"output_{i:03d}.jpg"
            image.save(image_path)

            # Get caption from vision model
            caption = get_image_caption(image_path)

            # Append caption to the text file
            caption_file.write(f"Caption for page {i+1}:\n{caption}\n\n")

            # Remove the image file after processing
            os.remove(image_path)

    print("Captions saved to captions.txt")

    # Read the captions file
    with open("temp/captions.txt", "r", encoding="utf-8") as caption_file:
        text = caption_file.read()

    return text