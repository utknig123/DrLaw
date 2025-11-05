# DrLAW - AI Legal Assistant

## Project Theory and Architecture

DrLAW is an innovative AI-powered legal assistance system designed to make legal information more accessible to the Indian population. The project combines several advanced technologies to provide accurate, context-aware legal advice in multiple Indian languages.

### Core Components

1. **Retrieval Augmented Generation (RAG) System**
   - Uses a hybrid approach combining retrieval and generation
   - Indexes legal documents for accurate context retrieval
   - Employs semantic search for finding relevant legal information
   - Components:
     - Document Processor (PDF extraction)
     - Text Chunker (semantic splitting)
     - Embedding Model (sentence-transformers)
     - Vector Database (FAISS)
     - LLM Integration (Google Gemini)

2. **Multilingual Support**
   - Supports 11 Indian languages:
     - Hindi, Bengali, Telugu, Marathi, Tamil
     - Gujarati, Kannada, Malayalam, Punjabi, Odia
   - Uses Google Translate API for accurate translations
   - Maintains legal terminology accuracy

3. **Knowledge Base**
   - Processed legal documents including:
     - Indian Constitution
     - Major Acts and Laws
     - Legal Precedents
     - Legal Procedures
   - Organized in semantic chunks for efficient retrieval
   - Regularly updated with new legal information

### Technical Implementation

1. **Backend Architecture (Flask)**
   ```
   /backend
   ├── app.py              # Main application logic
   ├── static/            # Static assets
   ├── templates/         # HTML templates
   ├── storage/           # Vector DB and chunks
   └── requirements.txt   # Dependencies
   ```

2. **Frontend Structure**
   ```
   /frontend
   ├── index.html        # Main interface
   ├── login.html        # Authentication
   ├── front.html        # Landing page
   └── config.js         # Configuration
   ```

3. **Database Schema (Supabase)**
   ```sql
   -- Users table
   CREATE TABLE users (
       user_id SERIAL PRIMARY KEY,
       username VARCHAR(255) NOT NULL,
       email VARCHAR(255) UNIQUE NOT NULL,
       password_hash VARCHAR(255),
       google_id VARCHAR(255) UNIQUE
   );

   -- Chats table
   CREATE TABLE chats (
       chat_id SERIAL PRIMARY KEY,
       user_id INTEGER NOT NULL REFERENCES users(user_id),
       question TEXT NOT NULL,
       answer TEXT NOT NULL,
       timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
   );
   ```

### RAG System Workflow

1. **Document Processing**
   - PDFs are processed and split into semantic chunks
   - Each chunk is embedded using sentence-transformers
   - Embeddings are stored in FAISS index
   - Metadata maintained for source tracking

2. **Query Processing**
   - User question is embedded
   - Similar chunks retrieved from FAISS
   - Context assembled from relevant chunks
   - Prompt constructed with legal format

3. **Response Generation**
   - Gemini API generates detailed response
   - Response structured with:
     - Legal analysis
     - Applicable laws
     - Required documentation
     - Step-by-step guidance

4. **Translation Flow**
   - Original response in English
   - Translation to requested language
   - Format preservation in translation

## Setup and Configuration

### Environment Variables
```env
FLASK_SECRET_KEY=<secret>
GEMINI_API_KEY=<your-key>
SUPABASE_URL=<url>
SUPABASE_KEY=<key>
GOOGLE_CLIENT_ID=<id>
GOOGLE_CLIENT_SECRET=<secret>
```

### Installation

1. Clone and setup:
```bash
git clone https://github.com/utknig123/DRLAW.git
cd DRLAW
```

2. Backend setup:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Start server:
```bash
flask run
```

## System Requirements

- Python 3.8+
- 4GB+ RAM (for embeddings)
- Storage for vector database
- Internet connection for APIs

## Security Considerations

1. **Authentication**
   - Session-based auth
   - Google OAuth integration
   - Password hashing
   - CORS protection

2. **Data Protection**
   - Encrypted storage
   - Secure API calls
   - Rate limiting
   - Input sanitization

## Future Enhancements

1. **Technical Improvements**
   - Real-time document updates
   - Improved semantic search
   - Caching system
   - Load balancing

2. **Feature Additions**
   - Document upload interface
   - Expert verification system
   - Legal form generation
   - Citation system

## Troubleshooting

Common issues and solutions:
1. Missing API keys
2. Database connection errors
3. Memory issues with embeddings
4. Session management problems
5. CORS configuration issues

## License

This project is licensed under the MIT License.
