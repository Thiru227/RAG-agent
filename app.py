"""
RTC Scholar - RAG-based AI Assistant (IMPROVED VERSION)
========================================================
Enhanced keyword matching with phrase detection and entity recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import re
from typing import List
from collections import Counter
import time

# Import knowledge base from separate file
from knowledge_base import KNOWLEDGE_BASE

# ============================================
# FLASK APP SETUP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# IMPROVED VECTOR DB WITH BETTER SEARCH
# ============================================
class ImprovedVectorDB:
    """Enhanced keyword-based retrieval with phrase matching and fuzzy search"""
    
    def __init__(self):
        self.documents = []
        print(f"✓ ImprovedVectorDB initialized")
    
    def add_documents(self, docs: List[str]):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
        print(f"✓ Loaded {len(docs)} documents into VectorDB")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = ' '.join(text.split())
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords, removing common stop words"""
        stop_words = {
            'what', 'is', 'the', 'who', 'where', 'when', 'how', 'are', 'do', 
            'does', 'about', 'tell', 'me', 'can', 'you', 'a', 'an', 'and', 
            'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        normalized = self.normalize_text(text)
        words = [w for w in normalized.split() if w not in stop_words and len(w) > 2]
        return words
    
    def calculate_relevance_score(self, query: str, doc: str) -> float:
        """Calculate relevance score using multiple factors"""
        query_normalized = self.normalize_text(query)
        doc_normalized = self.normalize_text(doc)
        
        score = 0.0
        
        # Factor 1: Exact phrase match (highest weight)
        if query_normalized in doc_normalized:
            score += 100
            position = doc_normalized.find(query_normalized)
            score += (100 - min(position, 100)) / 10
        
        # Factor 2: Keyword overlap
        query_keywords = self.extract_keywords(query)
        doc_keywords = self.extract_keywords(doc)
        
        if query_keywords:
            query_counter = Counter(query_keywords)
            doc_counter = Counter(doc_keywords)
            
            overlap = 0
            for keyword in query_counter:
                if keyword in doc_counter:
                    overlap += min(query_counter[keyword], doc_counter[keyword])
            
            keyword_score = (overlap / len(query_keywords)) * 50
            score += keyword_score
        
        # Factor 3: Role-based matching
        role_keywords = {
            'principal': ['principal', 'nagaraj', 'balakrishnan'],
            'ceo': ['ceo', 'manickam', 'chief', 'executive'],
            'chairman': ['chairman', 'sendhil', 'madan'],
            'vice principal': ['vice', 'principal', 'geetha'],
            'placement': ['placement', 'senthilkumar', 'career'],
            'dean': ['dean', 'saravanan', 'research', 'innovation'],
            'training': ['training', 'priya', 'ramachandran', 'career']
        }
        
        for role, related_terms in role_keywords.items():
            if any(term in query_normalized for term in [role]):
                if any(term in doc_normalized for term in related_terms):
                    score += 30
        
        # Factor 4: Name matching
        query_words = query_normalized.split()
        for word in query_words:
            if len(word) > 5:
                if word in doc_normalized:
                    score += 25
                elif any(word[:5] in doc_word for doc_word in doc_normalized.split()):
                    score += 15
        
        return score
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Enhanced search with multiple ranking factors"""
        if not query or not self.documents:
            return []
        
        scored_docs = []
        for doc in self.documents:
            score = self.calculate_relevance_score(query, doc)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Debug logging
        print(f"🔍 Query: '{query}' | Found {len(scored_docs)} relevant docs")
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            print(f"  [{i+1}] Score: {score:.1f} | {doc[:60]}...")
        
        return [doc for _, doc in scored_docs[:top_k]]

# ============================================
# INITIALIZE VECTOR DB WITH KNOWLEDGE BASE
# ============================================
vector_db = ImprovedVectorDB()
vector_db.add_documents(KNOWLEDGE_BASE)
print(f"📚 Knowledge base ready: {len(KNOWLEDGE_BASE)} documents")

# ============================================
# OPENROUTER LLM CONFIGURATION
# ============================================
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

def call_llm(prompt: str, context: str) -> str:
    """Call OpenRouter API with RAG context"""
    
    if not OPENROUTER_API_KEY:
        return "⚠️ My AI brain isn't configured yet. Please set the OPENROUTER_API_KEY in Render."
    
    system_prompt = f"""You are RTC Scholar, the friendly AI assistant for Rathinam Technical Campus (RTC).

Your personality:
- Warm, helpful, and professional
- Concise and accurate
- Use emojis sparingly for friendliness

CRITICAL INSTRUCTION: Answer ONLY using information from this context:
{context}

Rules:
- Give direct, short answers based strictly on the context above
- If the context contains the answer, provide it clearly
- Do NOT say "no information" if the answer is in the context
- For names/titles, quote them exactly as shown in the context
- Be confident when the information is available
- End with a helpful offer: "Need anything else? 😊"
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
        "temperature": 0.3,  # Lower temperature for more consistent answers
        "max_tokens": 400
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        print(f"❌ LLM API Error: {str(e)}")
        return f"Sorry, I'm having trouble connecting right now. Please try again! 🔄"
    except (KeyError, IndexError) as e:
        print(f"❌ LLM Response Parse Error: {str(e)}")
        return f"Oops, something went wrong processing your request. 😅"

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - Basic info"""
    return jsonify({
        'service': 'RTC Scholar AI Assistant (Enhanced)',
        'status': 'running',
        'version': '2.0',
        'improvements': 'Better keyword matching, phrase detection, entity recognition',
        'endpoints': {
            'health': '/health',
            'webhook': '/webhook (POST)',
            'test': '/test (POST)',
            'documents': '/documents (GET)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'RTC Scholar AI (Enhanced)',
        'timestamp': time.time(),
        'documents_loaded': len(vector_db.documents),
        'api_configured': bool(OPENROUTER_API_KEY)
    }), 200

@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    """Main Dialogflow webhook endpoint with improved RAG"""
    
    try:
        req = request.get_json(silent=True, force=True)
        query_text = req.get('queryResult', {}).get('queryText', '')
        
        if not query_text:
            return jsonify({
                'fulfillmentText': 'Sorry, I didn\'t receive a valid query.'
            })
        
        print(f"📥 Query received: {query_text}")
        
        # STEP 1: Retrieve relevant documents with improved search
        relevant_docs = vector_db.search(query_text, top_k=4)  # Get more context
        
        if not relevant_docs:
            return jsonify({
                'fulfillmentText': 'Hmm, I couldn\'t find specific information about that. Could you rephrase your question? 🤔'
            })
        
        context = "\n\n".join(relevant_docs)
        print(f"✓ Retrieved {len(relevant_docs)} relevant documents")
        
        # STEP 2: Generate response using LLM
        response_text = call_llm(query_text, context)
        print(f"✓ Response generated: {response_text[:100]}...")
        
        # STEP 3: Return to Dialogflow
        return jsonify({
            'fulfillmentText': response_text,
            'source': 'webhook-enhanced'
        })
    
    except Exception as e:
        print(f"❌ Webhook Error: {str(e)}")
        return jsonify({
            'fulfillmentText': 'Sorry, something went wrong. Please try again! 🔄'
        }), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """Test endpoint with detailed debugging"""
    
    data = request.get_json()
    query = data.get('query', 'Who is the principal?')
    
    # Retrieve relevant documents
    relevant_docs = vector_db.search(query, top_k=3)
    context = "\n\n".join(relevant_docs) if relevant_docs else "No context found"
    
    # Generate response
    response = call_llm(query, context)
    
    return jsonify({
        'query': query,
        'retrieved_documents': relevant_docs,
        'num_documents_found': len(relevant_docs),
        'context_sent_to_llm': context,
        'response': response
    })

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in knowledge base"""
    return jsonify({
        'total': len(vector_db.documents),
        'documents': vector_db.documents[:10],  # First 10 for preview
        'note': 'Showing first 10 documents. Total available: ' + str(len(vector_db.documents))
    })

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting RTC Scholar (Enhanced) on port {port}")
    print(f"📊 Total documents in KB: {len(vector_db.documents)}")
    print(f"🔑 API Key configured: {bool(OPENROUTER_API_KEY)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
