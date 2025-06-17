"""
Euri AI Client for Healthcare Q&A Tool.

This module provides a client interface for Euri AI API using the official
euriai Python SDK with healthcare-specific optimizations.
"""

from typing import Any, Dict, List, Optional

from euriai import EuriaiClient
from loguru import logger

from ..config import get_settings


class EuriClientError(Exception):
    """Custom exception for Euri client errors."""
    pass


class EuriClient:
    """Client for Euri AI API using official SDK with healthcare optimizations."""
    
    def __init__(self):
        """Initialize the Euri client."""
        self.settings = get_settings()
        self.api_key = self.settings.euri_api_key
        
        # Initialize official Euri AI client
        self.client = EuriaiClient(
            api_key=self.api_key,
            model=self.settings.llm_model
        )
        
        logger.info(f"Initialized Euri AI client with model: {self.settings.llm_model}")
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Euri AI official SDK.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to settings)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            API response dictionary
        """
        # Use settings defaults if not provided
        temperature = temperature if temperature is not None else self.settings.temperature
        max_tokens = max_tokens or self.settings.max_tokens
        
        try:
            # Convert messages to prompt format for Euri AI
            prompt = self._convert_messages_to_prompt(messages)
            
            # Generate completion using official SDK
            response = self.client.generate_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Convert response to OpenAI-compatible format
            result = {
                "choices": [
                    {
                        "message": {
                            "content": response,
                            "role": "assistant"
                        }
                    }
                ]
            }
            
            logger.debug(f"Chat completion successful: {len(messages)} messages")
            return result
            
        except Exception as e:
            logger.error(f"Euri AI request failed: {e}")
            raise EuriClientError(f"API request failed: {e}")
    
    def generate_healthcare_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a healthcare-focused response.
        
        Args:
            query: User question
            context: Retrieved document context
            system_prompt: Optional system prompt override
            
        Returns:
            Generated response text
        """
        if system_prompt is None:
            system_prompt = self._get_healthcare_system_prompt()
        
        user_prompt = self._create_healthcare_prompt(query, context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.chat_completion(messages)
            
            if response.get('choices') and len(response['choices']) > 0:
                return response['choices'][0]['message']['content'].strip()
            else:
                raise EuriClientError("No response generated")
                
        except Exception as e:
            logger.error(f"Healthcare response generation failed: {e}")
            raise EuriClientError(f"Response generation failed: {e}")
    
    def _get_healthcare_system_prompt(self) -> str:
        """Get the healthcare-specific system prompt."""
        return """You are a knowledgeable healthcare research assistant specializing in evidence-based medicine. 
Your role is to provide accurate, well-researched answers based on peer-reviewed scientific literature.

Guidelines:
- Always base your answers on the provided research articles
- Be precise and factual, avoiding speculation
- Acknowledge limitations and uncertainties in the research
- Distinguish between established facts and preliminary findings
- Use clear, professional language accessible to healthcare professionals
- When discussing medical topics, always recommend consulting healthcare professionals for clinical decisions
- If the provided articles don't contain sufficient information, state this clearly
- Focus on intermittent fasting, obesity, diabetes, and metabolic disorders when relevant
- Cite specific studies or findings when possible"""
    
    def _create_healthcare_prompt(self, query: str, context: str) -> str:
        """Create a healthcare-focused prompt."""
        return f"""Based on the following research articles about healthcare and medical topics, please answer the user's question. 
Focus on providing evidence-based information from the provided sources.

Research Articles:
{context}

User Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the question with evidence from the research
2. Cites relevant findings from the provided articles
3. Mentions any limitations, conflicting findings, or areas of uncertainty
4. Provides practical implications when appropriate
5. Indicates if more research is needed
6. Recommends consulting healthcare professionals for clinical decisions

Answer:"""
    
    def generate_research_summary(
        self,
        topic: str,
        context: str
    ) -> str:
        """
        Generate a research summary for a given topic.
        
        Args:
            topic: Research topic
            context: Retrieved document context
            
        Returns:
            Generated research summary
        """
        system_prompt = """You are a research analyst specializing in healthcare literature reviews. 
Provide comprehensive, evidence-based summaries of research topics based on peer-reviewed literature."""
        
        user_prompt = f"""Based on the following research articles about {topic}, provide a comprehensive research summary that includes:

1. Overview of the current state of research
2. Key findings and areas of consensus
3. Areas of disagreement or uncertainty
4. Methodological considerations and study quality
5. Clinical implications and practical applications
6. Gaps in research and future directions
7. Recommendations for healthcare practice

Research Articles:
{context}

Please provide a structured, evidence-based summary:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.chat_completion(
                messages,
                temperature=0.1,  # Lower temperature for more consistent summaries
                max_tokens=800
            )
            
            if response.get('choices') and len(response['choices']) > 0:
                return response['choices'][0]['message']['content'].strip()
            else:
                raise EuriClientError("No summary generated")
                
        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            raise EuriClientError(f"Summary generation failed: {e}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to Euri AI API.
        
        Returns:
            True if connection is successful
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hello, this is a connection test."}
            ]
            
            response = self.chat_completion(
                test_messages,
                max_tokens=10
            )
            
            return bool(response.get('choices'))
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
