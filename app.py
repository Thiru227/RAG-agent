"""
RTC Scholar - RAG-based AI Assistant
=====================================
Dialogflow webhook with OpenRouter LLM integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from typing import List
import time

# Import knowledge base from separate file
from knowledge_base import KNOWLEDGE_BASE

# ============================================
# FLASK APP SETUP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# SIMPLE IN-MEMORY VECTOR DB
# ============================================
class SimpleVectorDB:
    """Lightweight keyword-based retrieval (no embeddings needed)"""
    
    def __init__(self):
        self.documents = []
        print(f"✓ VectorDB initialized")
    
    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
        print(f"✓ Loaded {len(docs)} documents into VectorDB")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Simple keyword matching retrieval"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score documents by keyword overlap
        scored_docs = []
        for doc in self.documents:
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            # Calculate overlap score
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored_docs.append((overlap, doc))
        
        # Sort by score and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:top_k]]

# ============================================
# INITIALIZE VECTOR DB WITH KNOWLEDGE BASE
# ============================================
vector_db = SimpleVectorDB()
vector_db.add_documents(KNOWLEDGE_BASE)
print(f"📚 Knowledge base ready: {len(KNOWLEDGE_BASE)} documents")

# ============================================
# OPENROUTER LLM CONFIGURATION
# ============================================
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

def call_llm(prompt: str, context: str) -> str:
    """
    Call OpenRouter API with RAG context
    
    Args:
        prompt: User's question
        context: Retrieved documents from vector DB
        
    Returns:
        LLM generated response
    """
    
    if not OPENROUTER_API_KEY:
        return "⚠️ My AI brain isn't configured yet. Please set the OPENROUTER_API_KEY in Render."
    
    system_prompt = f"""
You are RTC Scholar, the friendly and knowledgeable AI assistant for Rathinam Technical Campus (RTC).  

Your personality:
- Warm, supportive, and student-friendly 🎓✨  
- Use emojis sparingly to sound natural (e.g., 📚🚀)  
- Be concise and clear — short answers only.  

Knowledge Use:
- Answer strictly and accurately based on the provided context from RTC's knowledge base:
{context}

Guidelines:
- Provide **short, precise answers** directly derived from the vector database context.  
- Do NOT add extra info or assumptions beyond the context.  
- If asked about admissions, facilities, or academics — respond briefly but correctly.  
- Maintain a positive, encouraging tone.  
- End with a short friendly offer to help further (e.g., "Would you like to know more? 😊").
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com",
        "X-Title": "DialogflowRAG"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM API Error: {str(e)}")
        return f"Error calling LLM: {str(e)}"
    except (KeyError, IndexError) as e:
        print(f"❌ LLM Response Parse Error: {str(e)}")
        return f"Error parsing LLM response: {str(e)}"

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - Basic info"""
    return jsonify({
        'service': 'RTC Scholar AI Assistant',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'webhook': '/webhook (POST)',
            'test': '/test (POST)',
            'documents': '/documents (GET)',
            'add_document': '/add-document (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring services (UptimeRobot, etc.)
    
    Returns lightweight status without loading heavy resources
    """
    return jsonify({
        'status': 'healthy',
        'service': 'RTC Scholar AI',
        'timestamp': time.time(),
        'documents_loaded': len(vector_db.documents),
        'api_configured': bool(OPENROUTER_API_KEY)
    }), 200

@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    """
    Main Dialogflow webhook endpoint
    
    Data Flow:
    1. Receives query from Dialogflow
    2. Searches vector DB for relevant documents (RAG Retrieval)
    3. Sends context + query to LLM (RAG Generation)
    4. Returns formatted response to Dialogflow
    """
    
    try:
        req = request.get_json(silent=True, force=True)
        query_text = req.get('queryResult', {}).get('queryText', '')
        
        if not query_text:
            return jsonify({
                'fulfillmentText': 'Sorry, I didn\'t receive a valid query.'
            })
        
        print(f"📥 Query received: {query_text}")
        
        # STEP 1: Retrieve relevant documents (RAG - Retrieval)
        relevant_docs = vector_db.search(query_text, top_k=3)
        
        if not relevant_docs:
            context = "No relevant information found in knowledge base."
            print(f"⚠️ No relevant documents found")
        else:
            context = "\n\n".join(relevant_docs)
            print(f"✓ Retrieved {len(relevant_docs)} relevant documents")
        
        # STEP 2: Generate response using LLM (RAG - Generation)
        response_text = call_llm(query_text, context)
        print(f"✓ Response generated")
        
        # STEP 3: Return to Dialogflow
        return jsonify({
            'fulfillmentText': response_text,
            'source': 'webhook'
        })
    
    except Exception as e:
        print(f"❌ Webhook Error: {str(e)}")
        return jsonify({
            'fulfillmentText': f'Error processing request: {str(e)}'
        }), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """
    Test endpoint to verify RAG pipeline
    
    Example request:
    POST /test
    {
        "query": "What programs does RTC offer?"
    }
    """
    
    data = request.get_json()
    query = data.get('query', 'What is RTC?')
    
    # Retrieve relevant documents
    relevant_docs = vector_db.search(query, top_k=3)
    context = "\n\n".join(relevant_docs) if relevant_docs else "No context found"
    
    # Generate response
    response = call_llm(query, context)
    
    return jsonify({
        'query': query,
        'retrieved_documents': relevant_docs,
        'response': response,
        'num_documents_found': len(relevant_docs)
    })

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in knowledge base"""
    return jsonify({
        'total': len(vector_db.documents),
        'documents': vector_db.documents
    })

@app.route('/add-document', methods=['POST'])
def add_document():
    """
    Add new documents to knowledge base dynamically
    
    Example request:
    POST /add-document
    {
        "documents": ["New info about RTC", "Another document"]
    }
    """
    
    data = request.get_json()
    documents = data.get('documents', [])
    
    if not documents or not isinstance(documents, list):
        return jsonify({'error': 'Please provide a list of documents'}), 400
    
    vector_db.add_documents(documents)
    print(f"✓ Added {len(documents)} new documents")
    
    return jsonify({
        'message': f'Added {len(documents)} documents',
        'total_documents': len(vector_db.documents)
    })

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting RTC Scholar on port {port}")
    print(f"📊 Total documents in KB: {len(vector_db.documents)}")
    print(f"🔑 API Key configured: {bool(OPENROUTER_API_KEY)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
