from flask import Flask, request, jsonify
import os
import requests
from flask_cors import CORS
from typing import List, Dict
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
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
    KNOWLEDGE_BASE = [
    # About RTC
    "Rathinam Technical Campus (RTC) is a premier educational institution located in Coimbatore, Tamil Nadu, India. We are committed to providing quality technical education and fostering innovation.",
    "RTC was established with the vision of creating industry-ready professionals through world-class education and practical training.",
    "The campus is spread across acres of lush green environment, providing an ideal atmosphere for learning and personal growth.",
    
    # Academics
    "RTC offers undergraduate programs (B.E/B.Tech) in Computer Science Engineering, Electronics and Communication Engineering, Mechanical Engineering, Civil Engineering, and Information Technology.",
    "We also offer postgraduate programs (M.E/M.Tech) in various specializations including Computer Science, VLSI Design, Structural Engineering, and more.",
    "The academic curriculum is regularly updated to meet industry standards and includes hands-on projects, internships, and industry collaborations.",
    "RTC follows a semester system with continuous internal assessment and end-semester examinations.",
    
    # Admissions
    "Admissions to RTC are based on merit in qualifying examinations and entrance test scores (Tamil Nadu Engineering Admissions - TNEA for B.E/B.Tech).",
    "For B.E/B.Tech programs, candidates must have completed 10+2 with Physics, Chemistry, and Mathematics with a minimum of 50% aggregate marks.",
    "M.E/M.Tech admissions require a valid GATE score or TANCET score along with a relevant undergraduate degree.",
    "The admission process typically begins in May-June for the academic year starting in August.",
    "Application forms are available online on the official RTC website. The application fee is Rs. 500 for general category and Rs. 250 for reserved categories.",
    
    # Facilities
    "RTC boasts state-of-the-art laboratories equipped with the latest technology and equipment for all engineering disciplines.",
    "The campus has a well-stocked central library with over 50,000 books, journals, e-resources, and digital library access.",
    "Hostel facilities are available separately for boys and girls with 24/7 security, mess facilities, and recreational areas.",
    "The campus features modern sports facilities including basketball court, volleyball court, cricket ground, and indoor games facilities.",
    "RTC has high-speed Wi-Fi connectivity across the entire campus, enabling students to access online resources anytime.",
    
    # Placements
    "RTC has an excellent placement record with 85%+ students getting placed every year in top companies.",
    "Leading companies like TCS, Infosys, Wipro, Cognizant, Accenture, Amazon, and many more recruit from RTC campus.",
    "The Training and Placement Cell conducts regular skill development programs, mock interviews, and aptitude training sessions.",
    "The average placement package ranges from 3.5 to 6 LPA, with highest packages going up to 12 LPA for exceptional performers.",
    
    # Faculty
    "RTC has a team of highly qualified and experienced faculty members with Ph.D. and M.E/M.Tech degrees from premier institutions.",
    "Faculty members actively engage in research activities and have published numerous papers in reputed international journals.",
    
    # Student Life
    "RTC has numerous student clubs including coding club, robotics club, cultural club, NSS, and entrepreneurship cell.",
    "Annual technical festival 'TechnoRTC' and cultural festival 'Rhythmica' are organized with participation from colleges across Tamil Nadu.",
    "Students actively participate in hackathons, coding competitions, project expos, and other inter-college events.",
    
    # Infrastructure
    "The campus features modern classrooms with smart boards, projectors, and audio-visual aids.",
    "RTC has dedicated computer centers with 500+ systems and licensed software for students.",
    "The campus cafeteria provides hygienic and nutritious food at affordable prices.",
    "24/7 medical facilities with a qualified doctor and ambulance service are available on campus.",
    
    # Contact Information
    "RTC is located at Eachanari, Coimbatore, Tamil Nadu - 641021, easily accessible from Coimbatore city.",
    "For admissions enquiry, contact: +91-422-2608800 or email: admissions@rtc.edu.in",
    "For general enquiry, email: info@rtc.edu.in or visit our website: www.rtc.edu.in",
    
    # Fees
    "The tuition fee for B.E/B.Tech programs is approximately Rs. 60,000 per semester (subject to change).",
    "Hostel fees including mess charges are approximately Rs. 35,000 per semester.",
    "Various scholarships are available for meritorious and economically disadvantaged students."
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
        return "⚠️ My AI brain isn't configured yet. Please set the OPENROUTER_API_KEY in Render."
    
    system_prompt = f"""You are RTC Scholar, the friendly AI assistant for Rathinam Technical Campus (RTC). 

Your personality:
- Helpful, enthusiastic, and knowledgeable about RTC
- Use emojis occasionally to be friendly (🎓 📚 ✨ 🚀)
- Be encouraging and supportive to students
- Address students in a warm, approachable manner

Use the following context from RTC's knowledge base to answer questions accurately:
{context}

Guidelines:
- Answer based ONLY on the provided context
- Be concise but informative
- Always maintain a positive, supportive tone
- If asked about admissions, facilities, or academics - provide detailed info from context
- End responses with an offer to help further"""

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
        'service': 'RTC Scholar AI Assistant',
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


