# Document Intelligence Platform

A comprehensive solution for extracting text from various document formats, processing it with AI, generating knowledge graphs, and providing a conversational interface for document queries.

## Features

- **Multi-format Document Processing**: Support for PDF, DOCX, DOC, TXT, CSV, and XLSX files
- **Vector Search**: Efficient document retrieval using Elasticsearch and vector embeddings
- **Knowledge Graph Generation**: Visual representation of relationships within documents
- **AI-powered Q&A**: Answer questions about documents using LLMs
- **Document Summarization**: Generate comprehensive document summaries
- **WhatsApp Integration**: Chat with your documents via WhatsApp
- **Multi-language Support**: Interface in various languages using translation APIs
- **Database Integration**: Store and query structured data from CSV/Excel files
- **User Management**: Free trial and paid subscription tiers

## Project Structure

```
.
├── app.py                      # Main Flask application
├── controllers/                # Business logic controllers
│   ├── __init__.py
│   ├── ask.py                  # Document query processing
│   ├── database.py             # Database operations
│   ├── doc_summary.py          # Document summarization
│   └── upload.py               # File upload processing
├── graph.py                    # Knowledge graph generation
├── local/
│   └── cleanup_sessions.py     # Session cleanup utility
├── requirements.txt            # Project dependencies
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── extractText.py          # Text extraction from files
└── webhook/                    # WhatsApp integration
    ├── __init__.py
    ├── webhook.py              # Webhook routes
    ├── decorators/             # Route decorators
    │   ├── __init__.py
    │   └── security.py         # Webhook verification
    ├── services/               # Service modules
    │   ├── __init__.py
    │   ├── db_operations.py    # Message tracking
    │   ├── demo_service.py     # Demo response generation
    │   ├── openai_service.py   # OpenAI integration
    │   └── sessionInfo.py      # Session management
    └── utils/                  # Webhook utilities
        ├── __init__.py
        └── whatsapp_utils.py   # WhatsApp API utilities
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/document-intelligence.git
   cd document-intelligence
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```
   ES_CLOUD_ID=your-elasticsearch-cloud-id
   ES_API_KEY=your-elasticsearch-api-key
   OPENAI_API_KEY=your-openai-api-key
   MYSQL_USERNAME=your-mysql-username
   MYSQL_PASSWORD=your-mysql-password
   ```

## Running the Application

### Flask API Server

```bash
python app.py
```

The Flask API will be available at `http://127.0.0.1:5000`.

### FastAPI Server

```bash
uvicorn custom_api:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI documentation will be available at `http://127.0.0.1:8000/docs`.

## API Endpoints

### Document Processing
- `POST /upload`: Upload documents
- `POST /add-upload`: Add documents to existing session
- `POST /freeTrial`: Process documents in free trial mode

### Document Querying
- `POST /ask`: Query documents
- `POST /trialAsk`: Query documents in free trial mode
- `POST /demo`: Demo mode for public transport queries

### Account Management
- `POST /updatepayment`: Update user payment plan
- `POST /check-payment-status`: Check remaining subscription time

### WhatsApp Webhook
- `GET|POST /api/webhook`: WhatsApp integration webhook

## External Integrations

- **Elasticsearch**: Vector storage and retrieval
- **OpenAI**: LLM for question answering and summarization
- **MongoDB**: User and session management
- **MySQL**: Structured data storage for CSV/Excel files
- **WhatsApp API**: Conversational interface

## Notes for Deployment

- For production, use a WSGI server like Gunicorn instead of the built-in Flask server
- Set up SSL certificates for secure HTTPS connections
- Configure proper database backups and monitoring
- Implement rate limiting for public endpoints

## Future Enhancements

- Integration with additional document formats
- Advanced OCR for image-based documents
- Multi-model support for different LLM backends
- Custom training for domain-specific document understanding
- Advanced analytics on document usage and query patterns

## License

This project is proprietary and confidential.