from flask import Flask, request, jsonify
import os
import requests
from typing import List, Dict
import json

app = Flask(__name__)

# ============================================
# SIMPLE IN-MEMORY VECTOR DB (No dependencies!)
# ============================================
class SimpleVectorDB:
    """Lightweight keyword-based retrieval (no embeddings needed)"""
    
    def __init__(self):
        self.documents = []
    
    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
    
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

# Initialize vector DB
vector_db = SimpleVectorDB()

# ============================================
# LOAD KNOWLEDGE BASE (Add your documents here)
# ============================================
KNOWLEDGE_BASE = [
    "Our company offers 24/7 customer support via email and phone.",
    "We provide free shipping on orders over $50 within the continental US.",
    "Returns are accepted within 30 days of purchase with original receipt.",
    "Our products come with a 1-year warranty covering manufacturing defects.",
    "We accept payments through credit cards, PayPal, and bank transfers.",
    "Business hours are Monday to Friday, 9 AM to 6 PM EST.",
    "You can track your order using the tracking number sent to your email.",
    "We offer discounts for bulk orders over 100 units.",
]

vector_db.add_documents(KNOWLEDGE_BASE)

# ============================================
# OPENROUTER API SETUP
# ============================================
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_llm(prompt: str, context: str) -> str:
    """Call OpenRouter API with RAG context"""
    
    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API key not configured"
    
    system_prompt = f"""You are a helpful customer service assistant. 
Use the following context to answer questions accurately:

CONTEXT:
{context}

If the answer is not in the context, politely say you don't have that information."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://render.com",
        "X-Title": "DialogflowRAG"
    }
    
    payload = {
        "model": "meta-llama/llama-3.2-3b-instruct:free",  # Free model
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
        return f"Error calling LLM: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing LLM response: {str(e)}"

# ============================================
# DIALOGFLOW WEBHOOK ENDPOINT
# ============================================
@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    """Handle Dialogflow webhook requests"""
    
    try:
        req = request.get_json(silent=True, force=True)
        
        # Extract query from Dialogflow request
        query_text = req.get('queryResult', {}).get('queryText', '')
        
        if not query_text:
            return jsonify({
                'fulfillmentText': 'Sorry, I didn\'t receive a valid query.'
            })
        
        # RAG Pipeline: Retrieve + Generate
        relevant_docs = vector_db.search(query_text, top_k=3)
        
        if not relevant_docs:
            context = "No relevant information found in knowledge base."
        else:
            context = "\n\n".join(relevant_docs)
        
        # Generate response using LLM
        response_text = call_llm(query_text, context)
        
        # Return Dialogflow response format
        return jsonify({
            'fulfillmentText': response_text,
            'source': 'webhook'
        })
    
    except Exception as e:
        return jsonify({
            'fulfillmentText': f'Error processing request: {str(e)}'
        }), 500

# ============================================
# HEALTH CHECK & TEST ENDPOINTS
# ============================================
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'service': 'RAG Agent',
        'documents_loaded': len(vector_db.documents),
        'api_key_configured': bool(OPENROUTER_API_KEY)
    })

@app.route('/test', methods=['POST'])
def test_endpoint():
    """Test endpoint to verify RAG pipeline"""
    
    data = request.get_json()
    query = data.get('query', 'What are your business hours?')
    
    # Retrieve relevant documents
    relevant_docs = vector_db.search(query, top_k=3)
    context = "\n\n".join(relevant_docs) if relevant_docs else "No context found"
    
    # Generate response
    response = call_llm(query, context)
    
    return jsonify({
        'query': query,
        'retrieved_documents': relevant_docs,
        'response': response
    })

# ============================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================
@app.route('/add-document', methods=['POST'])
def add_document():
    """Add new documents to knowledge base"""
    
    data = request.get_json()
    documents = data.get('documents', [])
    
    if not documents or not isinstance(documents, list):
        return jsonify({'error': 'Please provide a list of documents'}), 400
    
    vector_db.add_documents(documents)
    
    return jsonify({
        'message': f'Added {len(documents)} documents',
        'total_documents': len(vector_db.documents)
    })

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in knowledge base"""
    
    return jsonify({
        'total': len(vector_db.documents),
        'documents': vector_db.documents
    })

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)