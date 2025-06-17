#!/usr/bin/env python3
"""
Healthcare Q&A Tool - Demo Script

A demonstration script to showcase the capabilities of the Healthcare Q&A Tool.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_settings
from src.llm import EuriClient
from src.processing import DocumentProcessor
from src.qa_system import QAEngine


def test_euri_connection():
    """Test the Euri AI connection."""
    print("🔍 Testing Euri AI connection...")
    
    try:
        euri_client = EuriClient()
        if euri_client.test_connection():
            print("✅ Euri AI connection successful!")
            return True
        else:
            print("❌ Euri AI connection failed!")
            return False
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False


def demo_document_ingestion():
    """Demonstrate document ingestion."""
    print("\n📚 Demonstrating document ingestion...")
    
    try:
        processor = DocumentProcessor()
        
        # Small demo search
        search_term = "intermittent fasting obesity"
        max_results = 10
        
        print(f"🔍 Searching for: '{search_term}' (max {max_results} articles)")
        
        results = processor.search_and_ingest_pipeline(
            search_term=search_term,
            max_results=max_results,
            reset_collection=True
        )
        
        if results['success']:
            print("✅ Ingestion successful!")
            print(f"   📄 Articles found: {results['articles_found_in_pubmed']}")
            print(f"   ⚡ Articles processed: {results['total_articles_processed']}")
            print(f"   💾 Added to vector store: {results['articles_added_to_vector_store']}")
            return True
        else:
            print(f"❌ Ingestion failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return False


def demo_qa_system():
    """Demonstrate the Q&A system."""
    print("\n💬 Demonstrating Q&A system...")
    
    try:
        qa_engine = QAEngine()
        
        # Demo questions
        questions = [
            "What are the benefits of intermittent fasting for obesity?",
            "How effective is intermittent fasting for weight loss?",
            "What are the potential side effects of intermittent fasting?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n🤔 Question {i}: {question}")
            
            response = qa_engine.ask_question(
                question=question,
                n_documents=3,
                include_sources=False
            )
            
            print(f"🤖 Answer: {response['answer'][:200]}...")
            print(f"📊 Confidence: {response['confidence']:.2f}")
            print(f"📚 Sources used: {response['sources_count']}")
            
            if response.get('error'):
                print(f"⚠️ Warning: {response['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Q&A error: {e}")
        return False


def demo_analytics():
    """Demonstrate analytics capabilities."""
    print("\n📊 Demonstrating analytics...")
    
    try:
        qa_engine = QAEngine()
        insights = qa_engine.get_collection_insights()
        
        if insights.get('error'):
            print(f"❌ Analytics error: {insights['error']}")
            return False
        
        print("📈 Collection Statistics:")
        print(f"   📄 Total documents: {insights.get('total_documents', 0)}")
        print(f"   📚 Unique journals: {insights.get('unique_journals', 0)}")
        
        if insights.get('publication_years'):
            years = insights['publication_years']
            print(f"   📅 Publication years: {min(years)} - {max(years)}")
        
        if insights.get('document_analysis'):
            analysis = insights['document_analysis']
            
            if analysis.get('study_types'):
                print("   🔬 Study types:")
                for study_type, count in analysis['study_types'].items():
                    print(f"      - {study_type.replace('_', ' ').title()}: {count}")
            
            avg_relevance = analysis.get('average_relevance', 0)
            print(f"   ⭐ Average relevance: {avg_relevance:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics error: {e}")
        return False


def main():
    """Run the complete demo."""
    print("🏥 Healthcare Q&A Tool - Demo")
    print("=" * 50)
    
    # Check environment
    try:
        settings = get_settings()
        print(f"✅ Configuration loaded")
        print(f"   🔑 API Key: {'*' * 20}")
        print(f"   🏠 Base URL: {settings.euri_base_url}")
        print(f"   🤖 Model: {settings.llm_model}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Make sure you have a .env file with EURI_API_KEY set")
        return
    
    # Run demo steps
    steps = [
        ("Test Euri Connection", test_euri_connection),
        ("Document Ingestion", demo_document_ingestion),
        ("Q&A System", demo_qa_system),
        ("Analytics", demo_analytics)
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        success = step_func()
        results.append((step_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 Demo Summary:")
    
    for step_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {step_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All demo steps completed successfully!")
        print("🚀 You can now run the Streamlit app: python run_streamlit.py")
    else:
        print("\n⚠️ Some demo steps failed. Please check the errors above.")
        print("💡 Common issues:")
        print("   - Missing or invalid Euri API key")
        print("   - Network connectivity issues")
        print("   - Missing dependencies")


if __name__ == "__main__":
    main()
