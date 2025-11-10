#supabase working 
#google auth not working 
#version 1 final(deployable) - FULLY OPTIMIZED

from flask_cors import CORS
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from authlib.integrations.flask_client import OAuth
import os
import json
import nltk
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import nltk.data
import time
import requests
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from supabase import create_client, Client
from dotenv import load_dotenv
import threading

# Load environment variables FIRST
load_dotenv()

# Configure environment to prevent excessive downloads and memory issues
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer warnings

# Download NLTK data at startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# CORS configuration
cors_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
CORS(app, supports_credentials=True, origins=cors_origins)

# Supabase Configuration
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Get URLs from environment variables
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://drlaw.onrender.com')

# Initialize OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', ''),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', ''),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile',
                   'redirect_uri': f"{BACKEND_URL}/login/google/authorize"},
)

# Initialize the RAG system
API_KEY = os.environ.get("GEMINI_API_KEY") or None

# Translation function using direct API calls
def translate_text(text, dest_language='hi', src_language='en'):
    """Simple translation function using Google Translate API directly"""
    try:
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_language}&tl={dest_language}&dt=t&q={encoded_text}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            translation_data = response.json()
            if translation_data and len(translation_data) > 0:
                translated_text = translation_data[0][0][0]
                return translated_text
        
        return text
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return text

# Language codes and their names
LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'mr': 'Marathi',
    'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi', 'or': 'Odia'
}

class PDFProcessor:
    """Simple PDF text extractor"""
    def extract_text(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text

class TextChunker:
    """Splits text into chunks"""
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str) -> list:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class RAGSystem:
    """FULLY OPTIMIZED RAG system - minimal downloads & memory usage"""
    
    _instance = None
    
    def __new__(cls, api_key: str = None, supabase_client=None):  # ADD PARAMETERS HERE
        if cls._instance is None:
            cls._instance = super(RAGSystem, cls).__new__(cls)
            # Initialize only once when instance is created
            cls._instance._initialize(api_key, supabase_client)
        return cls._instance
    
    def _initialize(self, api_key: str = None, supabase_client=None):
        """Private initialization method"""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        print("🔄 Initializing OPTIMIZED RAG system...")
        
        self.api_key = api_key
        self.supabase = supabase_client
        self.pdf_bucket = "legal_pdfs"
        
        # Core components - loaded on demand
        self.pdf_processor = None
        self.chunker = None
        self.embedding_model = None
        self.llm = None
        self.index = None
        self.chunks = []
        
        # Status flags
        self.is_trained = False
        self.api_key_missing = False
        self.initialization_complete = False
        self.initialization_error = None
        
        # Storage setup with cleanup
        self.storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.storage_dir, "chunks_metadata.json")
        self.model_cache_dir = os.path.join(self.storage_dir, "model_cache")
        
        # Clean up any temporary files on startup
        self._cleanup_temp_files()
        
        # Configure Gemini only if API key exists
        if api_key:
            try:
                genai.configure(api_key=api_key)
                print("✅ Gemini configured successfully")
            except Exception as e:
                print(f"⚠️ Gemini config failed: {e}")
                self.api_key_missing = True
        else:
            self.api_key_missing = True
            print("⚠️ GEMINI_API_KEY not set")
        
        self._initialized = True
        print("✅ RAG system base initialization complete")    
    def _cleanup_temp_files(self):
        """Clean up temporary files to save storage"""
        try:
            # Remove any leftover PDF files
            for file in os.listdir(self.storage_dir):
                if file.endswith('.pdf'):
                    os.remove(os.path.join(self.storage_dir, file))
                    print(f"🧹 Cleaned up temporary file: {file}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def load_embedding_model(self):
        """Load ONLY the PyTorch model - prevent multiple format downloads"""
        if self.embedding_model is not None:
            return
            
        print("🔄 Loading sentence transformer model (optimized download)...")
        try:
            # Use your preferred smaller model
            model_name = 'paraphrase-albert-small-v2'
            
            # Create cache directory
            os.makedirs(self.model_cache_dir, exist_ok=True)
            
            # Load with cache to prevent re-downloads
            self.embedding_model = SentenceTransformer(
                model_name,
                cache_folder=self.model_cache_dir,
                device='cpu'
            )
            
            print(f"✅ Embedding model loaded: {model_name}")
            print(f"📏 Model dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            error_msg = f"❌ Failed to load embedding model: {e}"
            print(error_msg)
            self.initialization_error = error_msg
            raise
    
    def load_llm(self):
        """Load Gemini - API based, no model downloads"""
        if self.llm is not None:
            return
            
        if self.api_key_missing:
            raise RuntimeError("GEMINI_API_KEY is missing")
        
        try:
            self.llm = genai.GenerativeModel('gemini-2.0-flash')
            print("✅ LLM configured successfully (API-based)")
        except Exception as e:
            print(f"❌ Failed to configure LLM: {e}")
            raise
    
    def load_or_create_index(self):
        """Load or create FAISS index"""
        if self.index is not None:
            return
            
        print("🔄 Setting up FAISS index...")
        try:
            # Ensure embedding model is loaded first
            self.load_embedding_model()
            
            # Get dimension from model
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # Create simple index (memory efficient)
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            
            print(f"✅ FAISS index created with dimension: {dimension}")
        except Exception as e:
            print(f"❌ Failed to create FAISS index: {e}")
            self.initialization_error = f"FAISS index failed: {e}"
            raise
    
    def download_and_process_pdf(self, pdf_name: str) -> str:
        """Download and process PDF from Supabase with memory optimization"""
        temp_path = None
        try:
            print(f"📥 Processing {pdf_name}...")
            
            # Lazy load components
            if self.pdf_processor is None:
                self.pdf_processor = PDFProcessor()
            if self.chunker is None:
                self.chunker = TextChunker()
            
            # Ensure index is ready
            self.load_or_create_index()
            
            # Check if already processed
            if any(pdf_name in chunk.get('source', '') for chunk in self.chunks):
                return f"PDF {pdf_name} was already processed."
            
            # Download PDF from Supabase
            pdf_data = self.supabase.storage.from_(self.pdf_bucket).download(pdf_name)
            
            # Save to temporary file
            temp_path = os.path.join(self.storage_dir, pdf_name)
            with open(temp_path, 'wb') as f:
                f.write(pdf_data)
            
            # Process the PDF in memory-efficient way
            text = self.pdf_processor.extract_text(temp_path)
            new_chunks = self.chunker.chunk_text(text)
            
            # Process chunks in very small batches to save memory
            batch_size = 2  # Even smaller batches
            total_processed = 0
            
            for i in range(0, len(new_chunks), batch_size):
                batch_chunks = new_chunks[i:i+batch_size]
                
                # Encode batch
                embeddings = self.embedding_model.encode(batch_chunks)
                
                # Add to index
                self.index.add(embeddings.astype(np.float32))
                
                # Add metadata
                chunk_metadata = [{'text': chunk, 'source': pdf_name} for chunk in batch_chunks]
                self.chunks.extend(chunk_metadata)
                total_processed += len(batch_chunks)
                
                # Clear memory
                del embeddings
            
            self.is_trained = True
            
            # Save index and chunks
            self.save_index_and_chunks()
            
            result = f"✅ Successfully processed {total_processed} chunks from {pdf_name}"
            print(result)
            return result
            
        except Exception as e:
            error_msg = f"❌ Error processing PDF {pdf_name}: {str(e)}"
            print(error_msg)
            return error_msg
        finally:
            # Always clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def save_index_and_chunks(self):
        """Save FAISS index and chunks metadata"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            if self.chunks:
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.chunks, f)
        except Exception as e:
            print(f"⚠️ Error saving index: {e}")
    
    def load_saved_index(self):
        """Load saved index if exists"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                print("🔄 Loading saved index...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.chunks = json.load(f)
                self.is_trained = True
                print("✅ Saved index loaded successfully")
                return True
        except Exception as e:
            print(f"⚠️ Error loading saved index: {e}")
        return False
    
    def initialize_full_system(self):
        """Initialize the complete RAG system with error handling"""
        if self.initialization_complete:
            return True
            
        try:
            print("🚀 Starting FULL RAG system initialization...")
            
            # Step 1: Load embedding model (this downloads the model)
            self.load_embedding_model()
            
            # Step 2: Setup index
            self.load_or_create_index()
            
            # Step 3: Try to load saved index first (saves processing time)
            if self.load_saved_index():
                print("✅ Using saved index - no PDF processing needed")
            else:
                # Step 4: Process PDFs if no saved index
                print("📄 Processing PDFs for the first time...")
                pdf_files = ["constitution.pdf"]
                for pdf_file in pdf_files:
                    result = self.download_and_process_pdf(pdf_file)
                    print(f"📋 {result}")
            
            self.initialization_complete = True
            print("🎉 RAG system fully initialized and ready!")
            return True
            
        except Exception as e:
            self.initialization_error = f"Full initialization failed: {e}"
            print(f"❌ RAG system initialization failed: {e}")
            return False
    
    def query(self, question: str, conversation_history=None, language='en', top_k: int = 3) -> str:
        """Query the system with comprehensive error handling"""
        try:
            if self.api_key_missing:
                return ("❌ Error: Missing GEMINI_API_KEY. "
                        "Please set GEMINI_API_KEY in your environment variables.")
            
            # Ensure system is initialized
            if not self.initialization_complete:
                return "🔄 System is still initializing. Please try again in a moment."
            
            # Load LLM only when needed (no downloads here)
            self.load_llm()
            
            # Get relevant chunks with error handling
            query_embedding = self.embedding_model.encode([question])
            
            if len(self.chunks) == 0 or not self.is_trained:
                context = "No legal documents have been processed yet."
            else:
                try:
                    scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
                    context_chunks = []
                    for i in indices[0]:
                        if i < len(self.chunks):
                            context_chunks.append(self.chunks[i]['text'])
                    context = " ".join(context_chunks[:3])  # Limit context length
                except Exception as e:
                    print(f"⚠️ Error searching index: {e}")
                    context = "No legal documents available for search."
            
            # Format conversation history (limit to save memory)
            conversation_context = ""
            if conversation_history:
                conversation_context = "Previous conversation:\n"
                for message in conversation_history[-4:]:  # Last 4 messages only
                    role = "User" if message.get('role') == 'user' else "DrLAW"
                    conversation_context += f"{role}: {message.get('content')}\n"
            
            # Generate answer
            output_language = LANGUAGES.get(language, 'English')
            prompt = f"""You are DrLAW, a legal AI advisor. Provide detailed legal advice based on:

CONTEXT FROM LEGAL DOCUMENTS:
{context}

{conversation_context}

Current Question: {question}

Provide clear, structured legal advice in {output_language}. Use simple language and focus on practical guidance."""

            response = self.llm.generate_content(prompt)
            answer_text = response.text
            
            # Translate if needed
            if language != 'en':
                answer_text = translate_text(answer_text, language, 'en')
            
            return answer_text
            
        except Exception as e:
            error_msg = f"❌ Error generating answer: {str(e)}"
            print(error_msg)
            return error_msg

# Initialize RAG system immediately
print("🚀 Creating RAG system instance...")
rag = RAGSystem(api_key=API_KEY, supabase_client=supabase)

# Database helper functions
def get_user_by_email(email):
    try:
        response = supabase.table("users").select("*").eq("email", email).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        print(f"Error getting user by email: {e}")
        return None

def create_user(username, email, password_hash=None, google_id=None):
    try:
        user_data = {"username": username, "email": email}
        if password_hash:
            user_data["password_hash"] = password_hash
        if google_id:
            user_data["google_id"] = google_id
            
        response = supabase.table("users").insert(user_data).execute()
        return response.data[0]["user_id"]
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def save_chat(user_id, question, answer):
    if not user_id:
        raise ValueError("user_id is required")
    try:
        chat_data = {"user_id": user_id, "question": question, "answer": answer}
        supabase.table("chats").insert(chat_data).execute()
    except Exception as e:
        print(f"Error saving chat: {e}")
        raise

def get_chat_history(user_id):
    try:
        response = supabase.table("chats").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return []

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize RAG system in background with proper timing
def initialize_rag_background():
    """Initialize RAG system in background with proper error handling"""
    time.sleep(5)  # Wait longer for server to fully start
    print("🔄 Starting background RAG initialization...")
    try:
        success = rag.initialize_full_system()
        if success:
            print("✅ Background RAG initialization completed successfully")
        else:
            print(f"❌ Background RAG initialization failed: {rag.initialization_error}")
    except Exception as e:
        print(f"❌ Background RAG initialization crashed: {e}")

# Start background initialization
init_thread = threading.Thread(target=initialize_rag_background, daemon=True)
init_thread.start()

# Flask routes
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    return render_template('front.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = get_user_by_email(email)
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if get_user_by_email(email):
            flash('Email already registered', 'danger')
            return redirect(url_for('signup'))
        
        try:
            password_hash = generate_password_hash(password)
            user_id = create_user(username, email, password_hash)
            session['user_id'] = user_id
            session['username'] = username
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash('Error creating account', 'danger')
    
    return render_template('front.html')

@app.route('/login/google')
def google_login():
    redirect_uri = f"{BACKEND_URL}/login/google/authorize"
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/authorize')
def google_authorize():
    try:
        token = google.authorize_access_token()
        user_info = google.get('userinfo').json()
        
        response = supabase.table("users").select("*").eq("google_id", user_info['id']).execute()
        user = response.data[0] if response.data else None
        
        if not user:
            response = supabase.table("users").select("*").eq("email", user_info['email']).execute()
            user = response.data[0] if response.data else None
            
            if user:
                supabase.table("users").update({"google_id": user_info['id']}).eq("user_id", user['user_id']).execute()
            else:
                username = user_info.get('name', user_info['email'].split('@')[0])
                user_id = create_user(username=username, email=user_info['email'], google_id=user_info['id'])
                user = {'user_id': user_id, 'username': username}
        
        session['user_id'] = user['user_id']
        session['username'] = user['username']
        flash('Logged in with Google successfully!', 'success')
        return redirect(url_for('index'))
    
    except Exception as e:
        flash('Error logging in with Google', 'danger')
        return redirect(url_for('login'))

@app.route('/logout', methods=['GET','POST'])
def logout():
    session.clear()
    if request.method == 'POST':
        return jsonify({'status': 'ok'})
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
            
        data = request.json
        question = data.get('question')
        language = data.get('language', 'en')
        conversation_history = data.get('conversation_history', [])

        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Check if RAG system is ready
        if not rag.initialization_complete:
            return jsonify({
                'error': 'System is still initializing. Please try again in 30 seconds.',
                'status': 'initializing'
            }), 503
        
        # Process the question
        answer = rag.query(question, conversation_history, language)
        
        # Save to chat history
        save_chat(session.get('user_id'), question, answer)
        
        return jsonify({
            'question': question, 
            'answer': answer, 
            'language': language,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in /ask route: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/test', methods=['POST'])
def test_chat():
    """Public endpoint for testing chat without authentication"""
    try:
        data = request.json
        question = data.get('question', 'Test question')
        language = data.get('language', 'en')
        
        # Check if RAG system is ready
        if not rag.initialization_complete:
            return jsonify({
                'error': 'System initializing',
                'status': 'initializing'
            }), 503
        
        answer = rag.query(question, language=language)
        
        return jsonify({
            'question': question, 
            'answer': answer, 
            'language': language,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/history')
@login_required
def chat_history():
    history = get_chat_history(session['user_id'])
    return jsonify([dict(row) for row in history])

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    status_info = {
        'status': 'healthy',
        'rag_initialized': rag.initialization_complete,
        'rag_ready': rag.initialization_complete and rag.is_trained,
        'api_key_configured': not rag.api_key_missing,
        'chunks_loaded': len(rag.chunks) if rag.chunks else 0,
        'error': rag.initialization_error
    }
    
    status_code = 200 if rag.initialization_complete else 503
    return jsonify(status_info), status_code

@app.route('/status')
def status():
    """Detailed status endpoint"""
    return jsonify({
        'rag_system': {
            'initialized': rag.initialization_complete,
            'trained': rag.is_trained,
            'chunks_loaded': len(rag.chunks) if rag.chunks else 0,
            'api_key_configured': not rag.api_key_missing,
            'error': rag.initialization_error
        },
        'server': 'running'
    })

# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask server on port {port}...")
    print(f"🔧 RAG system status: {'Ready' if rag.initialization_complete else 'Initializing...'}")
    app.run(host='0.0.0.0', port=port, debug=False)