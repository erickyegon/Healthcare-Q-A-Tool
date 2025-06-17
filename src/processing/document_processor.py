"""
Document Processing and Ingestion Pipeline for Healthcare Q&A Tool.

This module provides comprehensive document processing capabilities,
including text cleaning, chunking, and metadata enrichment.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from ..config import get_settings
from ..data_retrieval import PubMedRetriever
from ..vector_store import ChromaManager


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class DocumentProcessor:
    """Comprehensive document processor for PubMed articles."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.settings = get_settings()
        self.pubmed_retriever = PubMedRetriever()
        self.chroma_manager = ChromaManager()
        
        logger.info("Initialized DocumentProcessor")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\,]{2,}', ',', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([\.,:;!?])\s*', r'\1 ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_key_information(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and enhance key information from article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Enhanced article with extracted information
        """
        enhanced_article = article.copy()
        
        # Clean title
        if enhanced_article.get('title'):
            enhanced_article['title'] = self.clean_text(enhanced_article['title'])
        
        # Process abstract
        abstract = enhanced_article.get('abstract', {})
        if isinstance(abstract, dict):
            cleaned_abstract = {}
            for section, content in abstract.items():
                if content and content != "No Abstract":
                    cleaned_abstract[section] = self.clean_text(content)
            enhanced_article['abstract'] = cleaned_abstract
        
        # Extract research focus areas
        enhanced_article['research_focus'] = self._identify_research_focus(enhanced_article)
        
        # Extract study type
        enhanced_article['study_type'] = self._identify_study_type(enhanced_article)
        
        # Calculate relevance score for healthcare topics
        enhanced_article['healthcare_relevance'] = self._calculate_healthcare_relevance(enhanced_article)
        
        return enhanced_article
    
    def _identify_research_focus(self, article: Dict[str, Any]) -> List[str]:
        """Identify research focus areas from article content."""
        focus_areas = []
        
        # Define focus keywords
        focus_keywords = {
            'intermittent_fasting': [
                'intermittent fasting', 'time-restricted eating', 'alternate day fasting',
                'periodic fasting', 'fasting intervention', 'caloric restriction'
            ],
            'obesity': [
                'obesity', 'overweight', 'weight loss', 'body mass index', 'bmi',
                'adiposity', 'weight management', 'bariatric'
            ],
            'diabetes': [
                'diabetes', 'diabetic', 'glucose', 'insulin', 'glycemic',
                'hyperglycemia', 'blood sugar', 'hba1c'
            ],
            'metabolic_disorders': [
                'metabolic syndrome', 'metabolism', 'metabolic disorder',
                'lipid profile', 'cholesterol', 'triglycerides'
            ],
            'cardiovascular': [
                'cardiovascular', 'heart disease', 'hypertension', 'blood pressure',
                'cardiac', 'coronary'
            ]
        }
        
        # Combine title, abstract, and keywords for analysis
        text_content = ""
        if article.get('title'):
            text_content += article['title'] + " "
        
        abstract = article.get('abstract', {})
        if isinstance(abstract, dict):
            text_content += " ".join(abstract.values()) + " "
        
        keywords = article.get('keywords', [])
        if keywords:
            text_content += " ".join(keywords)
        
        text_content = text_content.lower()
        
        # Check for focus areas
        for focus, keywords_list in focus_keywords.items():
            for keyword in keywords_list:
                if keyword.lower() in text_content:
                    focus_areas.append(focus)
                    break
        
        return list(set(focus_areas))  # Remove duplicates
    
    def _identify_study_type(self, article: Dict[str, Any]) -> str:
        """Identify the type of study from article content."""
        study_types = {
            'randomized_controlled_trial': [
                'randomized controlled trial', 'rct', 'randomized trial',
                'controlled trial', 'randomization'
            ],
            'systematic_review': [
                'systematic review', 'meta-analysis', 'systematic literature review'
            ],
            'cohort_study': [
                'cohort study', 'prospective study', 'longitudinal study'
            ],
            'case_control': [
                'case-control', 'case control study'
            ],
            'cross_sectional': [
                'cross-sectional', 'cross sectional study'
            ],
            'clinical_trial': [
                'clinical trial', 'intervention study'
            ],
            'review': [
                'review', 'literature review'
            ]
        }
        
        # Check article types first
        article_types = article.get('article_types', [])
        for article_type in article_types:
            article_type_lower = article_type.lower()
            if 'randomized controlled trial' in article_type_lower:
                return 'randomized_controlled_trial'
            elif 'systematic review' in article_type_lower:
                return 'systematic_review'
            elif 'meta-analysis' in article_type_lower:
                return 'systematic_review'
            elif 'clinical trial' in article_type_lower:
                return 'clinical_trial'
            elif 'review' in article_type_lower:
                return 'review'
        
        # Check title and abstract
        text_content = ""
        if article.get('title'):
            text_content += article['title'] + " "
        
        abstract = article.get('abstract', {})
        if isinstance(abstract, dict):
            text_content += " ".join(abstract.values())
        
        text_content = text_content.lower()
        
        for study_type, keywords in study_types.items():
            for keyword in keywords:
                if keyword in text_content:
                    return study_type
        
        return 'other'
    
    def _calculate_healthcare_relevance(self, article: Dict[str, Any]) -> float:
        """Calculate relevance score for healthcare topics (0-1)."""
        score = 0.0
        
        # Base score for having research focus areas
        focus_areas = article.get('research_focus', [])
        if focus_areas:
            score += 0.3 * len(focus_areas) / 5  # Max 5 focus areas
        
        # Bonus for high-quality study types
        study_type = article.get('study_type', '')
        quality_bonus = {
            'randomized_controlled_trial': 0.3,
            'systematic_review': 0.25,
            'clinical_trial': 0.2,
            'cohort_study': 0.15,
            'case_control': 0.1,
            'cross_sectional': 0.05
        }
        score += quality_bonus.get(study_type, 0)
        
        # Bonus for recent publications
        pub_date = article.get('publication_date', '')
        if pub_date:
            year = pub_date.split('-')[0]
            if year.isdigit():
                year_int = int(year)
                if year_int >= 2020:
                    score += 0.2
                elif year_int >= 2015:
                    score += 0.1
        
        # Bonus for having keywords
        if article.get('keywords'):
            score += 0.1
        
        # Bonus for having DOI (indicates peer review)
        if article.get('doi'):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def chunk_document(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document into chunks for better vector storage.

        Args:
            article: Article to chunk

        Returns:
            List of document chunks
        """
        chunks = []

        # Create main chunk with title and full abstract
        main_chunk = {
            'pmid': article.get('pmid'),
            'chunk_type': 'main',
            'title': article.get('title', ''),
            'content': article.get('full_text', ''),
            'metadata': {
                'pmid': article.get('pmid'),
                'title': article.get('title', ''),
                'journal': article.get('journal', ''),
                'authors': article.get('authors', ''),
                'publication_date': article.get('publication_date', ''),
                'doi': article.get('doi', ''),
                'keywords': str(article.get('keywords', [])),
                'article_types': str(article.get('article_types', [])),
                'research_focus': str(article.get('research_focus', [])),
                'study_type': article.get('study_type', ''),
                'healthcare_relevance': article.get('healthcare_relevance', 0.0),
                'chunk_type': 'main'
            }
        }
        chunks.append(main_chunk)

        # Create separate chunks for abstract sections if they're long
        abstract = article.get('abstract', {})
        if isinstance(abstract, dict):
            for section, content in abstract.items():
                if content and len(content) > self.settings.chunk_size:
                    section_chunk = {
                        'pmid': article.get('pmid'),
                        'chunk_type': f'abstract_{section.lower()}',
                        'title': f"{article.get('title', '')} - {section}",
                        'content': content,
                        'metadata': {
                            'pmid': article.get('pmid'),
                            'title': article.get('title', ''),
                            'journal': article.get('journal', ''),
                            'authors': article.get('authors', ''),
                            'publication_date': article.get('publication_date', ''),
                            'abstract_section': section,
                            'chunk_type': f'abstract_{section.lower()}',
                            'healthcare_relevance': article.get('healthcare_relevance', 0.0)
                        }
                    }
                    chunks.append(section_chunk)

        return chunks

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of articles with cleaning and enhancement.

        Args:
            articles: List of raw articles

        Returns:
            List of processed articles
        """
        processed_articles = []

        logger.info(f"Processing {len(articles)} articles")

        for article in tqdm(articles, desc="Processing articles"):
            try:
                # Extract and enhance information
                enhanced_article = self.extract_key_information(article)

                # Create full text representation
                enhanced_article['full_text'] = self._create_full_text(enhanced_article)

                processed_articles.append(enhanced_article)

            except Exception as e:
                pmid = article.get('pmid', 'unknown')
                logger.warning(f"Error processing article {pmid}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_articles)} articles")
        return processed_articles

    def _create_full_text(self, article: Dict[str, Any]) -> str:
        """Create comprehensive full text representation."""
        text_parts = []

        # Add title
        if article.get('title'):
            text_parts.append(f"Title: {article['title']}")

        # Add abstract sections
        abstract = article.get('abstract', {})
        if isinstance(abstract, dict):
            for section, content in abstract.items():
                if content and content != "No Abstract":
                    text_parts.append(f"{section}: {content}")

        # Add keywords
        keywords = article.get('keywords', [])
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")

        # Add research focus
        focus_areas = article.get('research_focus', [])
        if focus_areas:
            text_parts.append(f"Research Focus: {', '.join(focus_areas)}")

        return "\n\n".join(text_parts)

    def ingest_articles_to_vector_store(
        self,
        articles: List[Dict[str, Any]],
        reset_collection: bool = False
    ) -> Dict[str, Any]:
        """
        Complete pipeline to ingest articles into vector store.

        Args:
            articles: List of articles to ingest
            reset_collection: Whether to reset the collection first

        Returns:
            Ingestion results
        """
        logger.info(f"Starting ingestion pipeline for {len(articles)} articles")

        try:
            # Reset collection if requested
            if reset_collection:
                self.chroma_manager.reset_collection()
                logger.info("Reset vector collection")

            # Ensure collection exists
            self.chroma_manager.create_collection()

            # Process articles
            processed_articles = self.process_articles(articles)

            # Filter by relevance if needed
            high_relevance_articles = [
                article for article in processed_articles
                if article.get('healthcare_relevance', 0) > 0.3
            ]

            if len(high_relevance_articles) < len(processed_articles):
                logger.info(
                    f"Filtered to {len(high_relevance_articles)} high-relevance articles "
                    f"from {len(processed_articles)} total"
                )

            # Add to vector store
            added_count = self.chroma_manager.add_documents(high_relevance_articles)

            # Get collection stats
            stats = self.chroma_manager.get_collection_stats()

            results = {
                'total_articles_processed': len(processed_articles),
                'high_relevance_articles': len(high_relevance_articles),
                'articles_added_to_vector_store': added_count,
                'collection_stats': stats,
                'success': True
            }

            logger.info(f"Ingestion completed successfully: {results}")
            return results

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_articles_processed': 0,
                'articles_added_to_vector_store': 0
            }

    def search_and_ingest_pipeline(
        self,
        search_term: str,
        max_results: int = None,
        reset_collection: bool = False,
        **search_kwargs
    ) -> Dict[str, Any]:
        """
        Complete pipeline: search PubMed, process, and ingest articles.

        Args:
            search_term: PubMed search query
            max_results: Maximum articles to retrieve
            reset_collection: Whether to reset the collection

        Returns:
            Pipeline results
        """
        logger.info(f"Starting complete pipeline for search: '{search_term}'")

        try:
            # Search and fetch articles from PubMed
            articles = self.pubmed_retriever.search_and_fetch(
                search_term=search_term,
                max_results=max_results,
                **search_kwargs
            )

            if not articles:
                return {
                    'success': False,
                    'error': 'No articles found in PubMed search',
                    'search_term': search_term
                }

            # Ingest articles
            ingestion_results = self.ingest_articles_to_vector_store(
                articles=articles,
                reset_collection=reset_collection
            )

            # Combine results
            pipeline_results = {
                'search_term': search_term,
                'articles_found_in_pubmed': len(articles),
                **ingestion_results
            }

            logger.info(f"Complete pipeline finished: {pipeline_results}")
            return pipeline_results

        except Exception as e:
            logger.error(f"Complete pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'search_term': search_term
            }
