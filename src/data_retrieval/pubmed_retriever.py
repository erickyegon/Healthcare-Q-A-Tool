"""
Enhanced PubMed Retriever for Healthcare Q&A Tool.

This module provides robust PubMed data retrieval with improved error handling,
rate limiting, and comprehensive article processing.
"""

import time
from typing import Dict, List, Optional, Union
from xml.etree import ElementTree

import requests
from loguru import logger
from tqdm import tqdm

from ..config import get_settings


class PubMedAPIError(Exception):
    """Custom exception for PubMed API errors."""
    pass


class PubMedRetriever:
    """Enhanced PubMed article retriever with robust error handling and rate limiting."""
    
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self):
        """Initialize the PubMed retriever with settings."""
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'{self.settings.pubmed_tool_name}/1.0'
        })
        
        # Configure rate limiting
        self.last_request_time = 0
        self.rate_limit_delay = self.settings.pubmed_rate_limit_delay
        
        logger.info(f"Initialized PubMed retriever with rate limit: {self.rate_limit_delay}s")
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict, retries: int = None) -> requests.Response:
        """Make a robust HTTP request with retries and error handling."""
        if retries is None:
            retries = self.settings.pubmed_max_retries
        
        self._rate_limit()
        
        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    logger.error(f"Failed to make request after {retries + 1} attempts: {e}")
                    raise PubMedAPIError(f"API request failed: {e}")
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        raise PubMedAPIError("Unexpected error in request handling")
    
    def search_articles(
        self, 
        search_term: str, 
        max_results: int = None,
        date_range: Optional[str] = None,
        article_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Search for PubMed articles and return PMIDs.
        
        Args:
            search_term: Search query string
            max_results: Maximum number of results to return
            date_range: Date range filter (e.g., "2020:2023")
            article_types: List of article types to filter by
            
        Returns:
            List of PMIDs
        """
        if max_results is None:
            max_results = self.settings.max_articles_per_search
        
        logger.info(f"Searching PubMed for: '{search_term}' (max: {max_results})")
        
        # Build search query
        query = search_term
        if date_range:
            query += f" AND {date_range}[dp]"
        if article_types:
            type_filter = " OR ".join([f"{t}[pt]" for t in article_types])
            query += f" AND ({type_filter})"
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(100, max_results),  # API limit per request
            'retmode': 'xml',
            'usehistory': 'y'
        }
        
        if self.settings.pubmed_email:
            params['email'] = self.settings.pubmed_email
        
        pmid_list = []
        start = 0
        
        with tqdm(total=max_results, desc="Searching articles") as pbar:
            while len(pmid_list) < max_results:
                params['retstart'] = start
                
                try:
                    response = self._make_request(self.SEARCH_URL, params)
                    root = ElementTree.fromstring(response.content)
                    
                    # Check for errors
                    error_list = root.find(".//ErrorList")
                    if error_list is not None:
                        errors = [error.text for error in error_list.findall(".//PhraseNotFound")]
                        if errors:
                            logger.warning(f"PubMed search warnings: {errors}")
                    
                    # Extract PMIDs
                    ids = [id_elem.text for id_elem in root.findall(".//Id")]
                    if not ids:
                        logger.info("No more articles found")
                        break
                    
                    pmid_list.extend(ids)
                    pbar.update(len(ids))
                    
                    start += 100
                    
                except Exception as e:
                    logger.error(f"Error during search: {e}")
                    break
        
        result_pmids = pmid_list[:max_results]
        logger.info(f"Found {len(result_pmids)} articles")
        return result_pmids

    def fetch_articles(self, pmid_list: List[str]) -> List[Dict]:
        """
        Fetch detailed article information for given PMIDs.

        Args:
            pmid_list: List of PubMed IDs

        Returns:
            List of article dictionaries with metadata
        """
        if not pmid_list:
            logger.warning("No PMIDs provided for fetching")
            return []

        logger.info(f"Fetching {len(pmid_list)} articles from PubMed")
        articles = []

        # Process in batches of 100 (API limit)
        batch_size = 100
        with tqdm(total=len(pmid_list), desc="Fetching articles") as pbar:
            for i in range(0, len(pmid_list), batch_size):
                batch_pmids = pmid_list[i:i + batch_size]

                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml'
                }

                if self.settings.pubmed_email:
                    fetch_params['email'] = self.settings.pubmed_email

                try:
                    response = self._make_request(self.FETCH_URL, fetch_params)
                    batch_articles = self._parse_articles(response.content)
                    articles.extend(batch_articles)
                    pbar.update(len(batch_pmids))

                except Exception as e:
                    logger.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                    continue

        logger.info(f"Successfully fetched {len(articles)} articles")
        return articles

    def _parse_articles(self, xml_content: bytes) -> List[Dict]:
        """Parse XML response and extract article information."""
        articles = []

        try:
            root = ElementTree.fromstring(xml_content)

            for article in root.findall(".//PubmedArticle"):
                try:
                    parsed_article = self._parse_single_article(article)
                    if parsed_article:
                        articles.append(parsed_article)
                except Exception as e:
                    pmid = self._safe_find_text(article, ".//PMID", "Unknown")
                    logger.warning(f"Error parsing article {pmid}: {e}")
                    continue

        except ElementTree.ParseError as e:
            logger.error(f"XML parsing error: {e}")

        return articles

    def _parse_single_article(self, article_elem) -> Optional[Dict]:
        """Parse a single PubmedArticle element."""
        # Extract PMID
        pmid = self._safe_find_text(article_elem, ".//PMID")
        if not pmid:
            return None

        # Extract title
        title = self._safe_find_text(article_elem, ".//ArticleTitle", "No Title")

        # Extract and process abstract
        abstract = self._extract_abstract(article_elem)

        # Extract journal information
        journal = self._safe_find_text(article_elem, ".//Journal/Title", "Unknown Journal")

        # Extract publication date
        pub_date = self._extract_publication_date(article_elem)

        # Extract authors
        authors = self._extract_authors(article_elem)

        # Extract keywords
        keywords = self._extract_keywords(article_elem)

        # Extract DOI
        doi = self._extract_doi(article_elem)

        # Extract article type
        article_types = self._extract_article_types(article_elem)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": authors,
            "publication_date": pub_date,
            "keywords": keywords,
            "doi": doi,
            "article_types": article_types,
            "full_text": self._create_full_text(title, abstract)
        }

    def _safe_find_text(self, element, xpath: str, default: str = "") -> str:
        """Safely extract text from XML element."""
        found = element.find(xpath)
        return found.text if found is not None and found.text else default

    def _extract_abstract(self, article_elem) -> Dict[str, str]:
        """Extract abstract sections."""
        abstract_sections = article_elem.findall(".//AbstractText")

        if not abstract_sections:
            return {"SUMMARY": "No Abstract"}

        abstract = {}
        for section in abstract_sections:
            label = section.attrib.get('Label', 'SUMMARY')
            text = section.text if section.text else ""
            if text:
                abstract[label] = text

        return abstract if abstract else {"SUMMARY": "No Abstract"}

    def _extract_publication_date(self, article_elem) -> str:
        """Extract publication date."""
        # Try different date formats
        year = self._safe_find_text(article_elem, ".//PubDate/Year")
        month = self._safe_find_text(article_elem, ".//PubDate/Month")
        day = self._safe_find_text(article_elem, ".//PubDate/Day")

        if year:
            date_parts = [year]
            if month:
                date_parts.append(month)
            if day:
                date_parts.append(day)
            return "-".join(date_parts)

        # Try alternative date format
        medline_date = self._safe_find_text(article_elem, ".//PubDate/MedlineDate")
        return medline_date if medline_date else "Unknown Year"

    def _extract_authors(self, article_elem) -> str:
        """Extract author information."""
        authors = []

        for author in article_elem.findall(".//Author"):
            first_name = self._safe_find_text(author, ".//ForeName")
            last_name = self._safe_find_text(author, ".//LastName")

            if first_name and last_name:
                authors.append(f"{first_name} {last_name}")
            elif last_name:
                authors.append(last_name)

        return ", ".join(authors) if authors else "No Authors"

    def _extract_keywords(self, article_elem) -> List[str]:
        """Extract keywords/MeSH terms."""
        keywords = []

        # Extract MeSH terms
        for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                keywords.append(mesh.text)

        # Extract author keywords
        for keyword in article_elem.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

        return list(set(keywords))  # Remove duplicates

    def _extract_doi(self, article_elem) -> Optional[str]:
        """Extract DOI if available."""
        for article_id in article_elem.findall(".//ArticleId"):
            if article_id.attrib.get('IdType') == 'doi':
                return article_id.text
        return None

    def _extract_article_types(self, article_elem) -> List[str]:
        """Extract publication types."""
        types = []
        for pub_type in article_elem.findall(".//PublicationType"):
            if pub_type.text:
                types.append(pub_type.text)
        return types

    def _create_full_text(self, title: str, abstract: Dict[str, str]) -> str:
        """Create full text representation for indexing."""
        text_parts = [title]

        for section, content in abstract.items():
            if content and content != "No Abstract":
                text_parts.append(f"{section}: {content}")

        return "\n\n".join(text_parts)

    def search_and_fetch(
        self,
        search_term: str,
        max_results: int = None,
        **search_kwargs
    ) -> List[Dict]:
        """
        Convenience method to search and fetch articles in one call.

        Args:
            search_term: Search query
            max_results: Maximum results to return
            **search_kwargs: Additional search parameters

        Returns:
            List of complete article dictionaries
        """
        pmids = self.search_articles(search_term, max_results, **search_kwargs)
        return self.fetch_articles(pmids)
