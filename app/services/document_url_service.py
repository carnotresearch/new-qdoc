"""
URL document processing service.

This module handles URL processing, web scraping, and content extraction
for the document upload system.
"""

import logging
import os
import threading
import json
import time
import requests
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.schema import Document

# Browser automation support
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from controllers.upload import store_vector
from controllers.doc_summary import create_abstractive_summary

# Configure logging
logger = logging.getLogger(__name__)

class DocumentUrlService:
    """Service for processing URLs and web scraping"""
    
    def __init__(self):
        self.driver = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }
    
    def _get_webdriver(self):
        """Initialize and return Chrome WebDriver instance optimized for JS-heavy sites"""
        if self.driver is None:
            try:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                
                self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                self.driver.implicitly_wait(10)
            except Exception as e:
                self.driver = None
        return self.driver
    
    def _cleanup_driver(self):
        """Clean up WebDriver instance"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except Exception as e:
                pass
    
    def __del__(self):
        """Destructor to ensure WebDriver cleanup"""
        self._cleanup_driver()
    
    def classify_url_files(self, file_list) -> List:
        """
        Classify files with .url extension as URL files
        
        Args:
            file_list: List of uploaded files/form data
            
        Returns:
            List of URL files
        """
        url_files = []
        for file in file_list:
            if file.filename and file.filename.endswith('.url'):
                url_files.append(file)
        return url_files
    
    def _extract_metadata(self, soup):
        """Extract metadata from HTML soup object"""
        metadata = {}
        for tag in soup.find_all("meta"):
            if tag.get("name"):
                metadata[tag.get("name")] = tag.get("content", "")
            elif tag.get("property"):
                metadata[tag.get("property")] = tag.get("content", "")
        return metadata
    
    def _try_static_scrape(self, url):
        """Try static scraping first (faster)"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            return None
    
    def _try_dynamic_scrape(self, url):
        """Enhanced Selenium scraping for JavaScript-heavy sites like Reddit"""
        try:
            driver = self._get_webdriver()
            if driver is None:
                return None
            
            driver.get(url)
            
            # Wait longer for JS-heavy sites and check for content
            for wait_time in [3, 5, 8]:
                time.sleep(wait_time)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                text_content = soup.get_text(strip=True)
                
                # If we have substantial content, use it
                if len(text_content) > 500:
                    return soup
            
            # Final attempt with maximum wait
            return BeautifulSoup(driver.page_source, 'html.parser')
        except Exception as e:
            return None
    
    def _is_valid_url(self, url, base_domain):
        """Check if URL is valid for crawling within the same domain"""
        try:
            parsed = urlparse(url)
            return parsed.netloc == base_domain and parsed.scheme in ['http', 'https']
        except:
            return False
    
    def crawl_website_with_limits(self, base_url: str, max_pages: int = 5, depth: int = 1) -> Dict[str, Any]:
        """
        Crawl website with depth and page limits like in your notebook
        
        Args:
            base_url: The URL to start crawling from
            max_pages: Maximum number of pages to scrape (default: 5)
            depth: Crawling depth (default: 1)
            
        Returns:
            Dict containing combined scraped content and metadata
        """
        try:
            visited = set()
            to_visit = [base_url]
            base_domain = urlparse(base_url).netloc
            scraped_pages = []
            total_content = ""
            
            pages_processed = 0
            while to_visit and pages_processed < max_pages:
                current_url = to_visit.pop(0)
                
                if current_url in visited:
                    continue
                
                visited.add(current_url)
                
                # Try static scraping first
                soup = self._try_static_scrape(current_url)
                used_selenium = False
                
                # Fallback to Selenium if needed
                if soup is None or len(soup.get_text(strip=True)) < 200:
                    soup = self._try_dynamic_scrape(current_url)
                    used_selenium = True
                
                if soup is None:
                    continue
                
                # Extract links for next depth level (only if depth > 1)
                if depth > 1 and pages_processed == 0:  # Only from the first page
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        full_url = urljoin(current_url, href)
                        
                        # Only add links from same domain
                        if (self._is_valid_url(full_url, base_domain) and 
                            full_url not in visited and 
                            full_url not in to_visit):
                            to_visit.append(full_url)
                
                # Clean and extract content
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    element.decompose()
                
                # Extract title
                title = soup.title.string if soup.title else urlparse(current_url).netloc
                title = title.strip() if title else urlparse(current_url).netloc
                
                # Extract main content
                content_selectors = [
                    'main', 'article', '[role="main"]', 
                    '.content', '.main-content', '.post-content',
                    '#content', '#main', 'body'
                ]
                
                main_content = None
                for selector in content_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                # Get text content
                if main_content:
                    text_content = main_content.get_text(separator='\n', strip=True)
                else:
                    text_content = soup.get_text(separator='\n', strip=True)
                
                # Clean content
                lines = text_content.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and len(line) > 3 and 
                        not line.lower() in ['home', 'about', 'contact', 'menu', 'search', 'login', 'logout'] and
                        not line.startswith(('©', '®', '™')) and
                        len(line.split()) > 1):
                        cleaned_lines.append(line)
                
                page_content = '\n'.join(cleaned_lines)
                
                if page_content:  # Only add if we got meaningful content
                    scraped_pages.append({
                        'url': current_url,
                        'title': title,
                        'content': page_content,
                        'used_selenium': used_selenium
                    })
                    
                    # Add to total content with page separator
                    total_content += f"\n\n=== PAGE: {title} ({current_url}) ===\n{page_content}"
                
                pages_processed += 1
            
            if not scraped_pages:
                return {
                    'success': False,
                    'url': base_url,
                    'error': 'No content could be extracted from any pages',
                    'content': ''
                }
            
            # Extract metadata from first page
            first_page_soup = self._try_static_scrape(base_url)
            if first_page_soup:
                metadata = self._extract_metadata(first_page_soup)
            else:
                metadata = {}
            
            description = (
                metadata.get('description') or 
                metadata.get('og:description') or 
                metadata.get('twitter:description') or 
                f"Content from {len(scraped_pages)} pages"
            )
            
            # Get overall title from first page
            overall_title = scraped_pages[0]['title'] if scraped_pages else urlparse(base_url).netloc
            
            return {
                'success': True,
                'url': base_url,
                'title': overall_title,
                'description': description,
                'content': total_content.strip(),
                'content_length': len(total_content),
                'pages_scraped': len(scraped_pages),
                'metadata': {
                    'author': metadata.get('author', ''),
                    'keywords': metadata.get('keywords', ''),
                    'pages_count': len(scraped_pages),
                    'extraction_method': 'crawling'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'url': base_url,
                'error': f"Crawling error: {str(e)}",
                'content': ''
            }
    
    def extract_url_from_file(self, url_file) -> str:
        """
        Extract URL from uploaded URL file
        
        Args:
            url_file: The uploaded URL file
            
        Returns:
            Extracted URL string
        """
        try:
            # Read the URL file content
            file_content = url_file.read().decode('utf-8')
            
            # Parse the URL from the file content
            try:
                # Try to parse as JSON first (from frontend structure)
                url_data = json.loads(file_content)
                url = url_data.get('url', '')
            except json.JSONDecodeError:
                # If not JSON, treat as plain URL
                url = file_content.strip()
            
            # Ensure URL has proper scheme
            if url and not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            return url
            
        except Exception as e:
            return ''
    
    def create_url_document(self, scraped_content: Dict[str, Any], filename: str) -> Document:
        """
        Create a LangChain Document object from scraped URL content
        
        Args:
            scraped_content: Dict containing scraped content and metadata
            filename: Name for the generated file
            
        Returns:
            LangChain Document object
        """
        # Enhanced document content with metadata
        metadata_info = scraped_content.get('metadata', {})
        document_content = f"""URL: {scraped_content['url']}
Title: {scraped_content['title']}
Description: {scraped_content['description']}
Author: {metadata_info.get('author', 'Unknown')}
Keywords: {metadata_info.get('keywords', 'None')}
Extraction Method: {metadata_info.get('extraction_method', 'static')}

Content:
{scraped_content['content']}"""
        
        # Create Document object with enhanced metadata
        doc = Document(
            page_content=document_content,
            metadata={
                'source': scraped_content['url'],
                'filename': filename,
                'title': scraped_content['title'],
                'page': 0,  # URLs are treated as single page documents (0-indexed like other docs)
                'url': scraped_content['url'],
                'extraction_method': metadata_info.get('extraction_method', 'static'),
                'used_selenium': scraped_content.get('used_selenium', False),
                'author': metadata_info.get('author', ''),
                'keywords': metadata_info.get('keywords', '')
            }
        )
        
        return doc
    
    def save_scraped_content(self, scraped_content: Dict[str, Any], user_session: str, url_index: int) -> str:
        """
        Save scraped content to a text file for backup/reference
        
        Args:
            scraped_content: Dict containing scraped content
            user_session: User session identifier  
            url_index: Index for unique filename generation
            
        Returns:
            Path to saved file
        """
        try:
            # Create session directory
            user_session_dir = os.path.join('users', user_session, 'files')
            os.makedirs(user_session_dir, exist_ok=True)
            
            # Create a text file name based on the URL
            safe_filename = f"url_{urlparse(scraped_content['url']).netloc.replace('.', '_')}_{url_index}.txt"
            document_path = os.path.join(user_session_dir, safe_filename)
            
            # Create enhanced document content
            metadata_info = scraped_content.get('metadata', {})
            document_content = f"""URL: {scraped_content['url']}
Title: {scraped_content['title']}
Description: {scraped_content['description']}
Author: {metadata_info.get('author', 'Unknown')}
Keywords: {metadata_info.get('keywords', 'None')}
Extraction Method: {metadata_info.get('extraction_method', 'static')}
Used Selenium: {scraped_content.get('used_selenium', False)}

Content:
{scraped_content['content']}"""
            
            # Write the scraped content to a file
            with open(document_path, 'w', encoding='utf-8') as f:
                f.write(document_content)
            
            return safe_filename
            
        except Exception as e:
            return ''
    
    def process_url_files(self, url_files: List, user_session: str, is_new_container: bool) -> Dict[str, Any]:
        """
        Process uploaded .url files by extracting URLs and scraping content
        
        Args:
            url_files: List of uploaded .url files
            user_session: User session identifier
            is_new_container: Whether to create a new container or add to existing
            
        Returns:
            Dict with processing results
        """
        if not url_files:
            return {"success": True, "files_processed": 0, "file_details": []}
        
        try:
            # Extract URLs from .url files
            urls = []
            for url_file in url_files:
                extracted_url = self.extract_url_from_file(url_file)
                if extracted_url:
                    urls.append(extracted_url)
            
            # Process the extracted URLs using the existing direct URL processing
            if urls:
                return self.process_urls_directly(urls, user_session, is_new_container)
            else:
                return {
                    "success": False,
                    "message": "No valid URLs could be extracted from .url files",
                    "files_processed": len(url_files),
                    "file_details": []
                }
                
        except Exception as e:
            logger.error(f"Error processing URL files: {e}")
            return {
                "success": False,
                "message": f"Error processing URL files: {str(e)}",
                "files_processed": len(url_files),
                "file_details": []
            }
    
    def process_urls_directly(self, urls: List[str], user_session: str, is_new_container: bool) -> Dict[str, Any]:
        """
        Process URLs directly without expecting .url files
        
        Args:
            urls: List of URL strings to process
            user_session: User session identifier
            is_new_container: Whether to create a new container or add to existing
            
        Returns:
            Dict with processing results
        """
        if not urls:
            return {"success": True, "files_processed": 0, "file_details": []}
        
        try:
            url_documents = []
            file_details = []
            file_infos = []
            successful_urls = 0
            
            for i, url in enumerate(urls):
                try:
                    # Validate URL
                    if not url.startswith(('http://', 'https://')):
                        file_details.append({
                            'filename': f'url_{i+1}',
                            'success': False,
                            'url': url,
                            'error': 'Invalid URL format'
                        })
                        continue
                    
                    # Crawl the website with limits (depth=1, max_pages=5)
                    scraped_result = self.crawl_website_with_limits(url, max_pages=5, depth=1)
                    
                    if scraped_result['success']:
                        # Save scraped content as backup file
                        safe_filename = self.save_scraped_content(scraped_result, user_session, i)
                        
                        # Create Document object for vector storage
                        doc = self.create_url_document(scraped_result, safe_filename)
                        url_documents.append(doc)
                        
                        # Create file info for tracking
                        file_infos.append({
                            'filename': safe_filename,
                            'success': True,
                            'url': url,
                            'title': scraped_result['title']
                        })
                        
                        successful_urls += 1
                        
                        file_details.append({
                            'filename': f'url_{i+1}',
                            'success': True,
                            'url': url,
                            'title': scraped_result['title'],
                            'content_length': scraped_result['content_length'],
                            'saved_as': safe_filename
                        })
                        
                    else:
                        file_details.append({
                            'filename': f'url_{i+1}',
                            'success': False,
                            'url': url,
                            'error': scraped_result['error']
                        })
                    
                except Exception as e:
                    file_details.append({
                        'filename': f'url_{i+1}',
                        'success': False,
                        'url': url,
                        'error': f"Processing error: {str(e)}"
                    })
            
            # Store documents in vector database if we have any
            if url_documents:
                # Store as vectors using existing system
                store_vector([url_documents], user_session, is_new_container, file_infos)
                
                # Create summaries asynchronously
                threading.Thread(target=create_abstractive_summary, args=(user_session,)).start()
            
            # Clean up WebDriver after processing
            self._cleanup_driver()
            
            return {
                "success": True,
                "files_processed": len(urls),
                "files_successful": successful_urls,
                "file_details": file_details
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing URLs: {str(e)}",
                "files_processed": len(urls),
                "file_details": []
            }

# Create a singleton instance
_document_url_service = None

def get_document_url_service() -> DocumentUrlService:
    """Get the document URL service singleton instance"""
    global _document_url_service
    if _document_url_service is None:
        _document_url_service = DocumentUrlService()
    return _document_url_service 