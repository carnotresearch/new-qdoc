# icarKno API Documentation

## Overview

The icarKno API provides intelligent document analysis capabilities including upload, processing, querying, and knowledge extraction. The API supports multiple document formats and offers both free trial and authenticated access.

**Base URL**: `https://qdocbackend.carnotresearch.com`

## Authentication

Most endpoints require JWT-based authentication. Include the token in request body:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

## Rate Limits

| User Type     | Limit                 | Scope       |
| ------------- | --------------------- | ----------- |
| Free Trial    | 10 queries            | Per session |
| Authenticated | Based on subscription | Per account |

---

## 📄 Document Management

### Upload Documents (New Session)

Create a new document session and upload files.

**Endpoint**: `POST /upload`  
**Content-Type**: `multipart/form-data`  
**Authentication**: Required

#### Request Parameters

```
token (string, required): JWT authentication token
sessionId (string, required): Unique session identifier
files (file[], required): Documents to upload
```

#### Supported Formats

- **Documents**: PDF, DOCX, DOC, TXT, PPTX
- **Data**: CSV, XLSX, XLS
- **Max Size**: 50MB per file

#### Response

```json
{
  "status": "ok",
  "message": "Files successfully uploaded.",
  "files_processed": 3,
  "files_successful": 3,
  "file_details": [
    {
      "filename": "document.pdf",
      "success": true,
      "page_count": 25,
      "chunk_count": 156
    }
  ]
}
```

#### Example

```bash
curl -X POST https://qdocbackend.carnotresearch.com/upload \
  -F "token=your-jwt-token" \
  -F "sessionId=session123" \
  -F "files=@document.pdf" \
  -F "files=@data.xlsx"
```

---

### Add Documents to Existing Session

Add additional documents to an existing session.

**Endpoint**: `POST /add-upload`  
**Content-Type**: `multipart/form-data`  
**Authentication**: Required

#### Request Parameters

Same as `/upload` but adds to existing session.

#### Response

Same format as `/upload`.

---

### Free Trial Upload

Upload documents without authentication (limited usage).

**Endpoint**: `POST /freeTrial`  
**Content-Type**: `multipart/form-data`  
**Rate Limit**: 10 per minute

#### Request Parameters

```
fingerprint (string, required): Unique browser fingerprint
files (file[], required): Documents to upload (no CSV/Excel in trial)
```

#### Response

```json
{
  "status": "ok",
  "message": "Files uploaded successfully",
  "files_processed": 1,
  "files_successful": 1
}
```

---

### Delete Session

Remove a session and all associated documents.

**Endpoint**: `POST /delete-container`  
**Authentication**: Required

#### Request

```json
{
  "token": "your-jwt-token",
  "sessionId": "session123"
}
```

---

## 🔍 Query & Analysis

### Query Documents (Authenticated)

Ask questions about uploaded documents with advanced reasoning.

**Endpoint**: `POST /ask`  
**Content-Type**: `application/json`  
**Authentication**: Required

#### Request Body

```json
{
  "token": "your-jwt-token",
  "sessionId": "session123",
  "message": "What are the key findings in the research?",
  "inputLanguage": 23,
  "outputLanguage": 23,
  "context": true,
  "hasCsvOrXlsx": false,
  "mode": "default",
  "filenames": ["specific_file.pdf"]
}
```

#### Parameters

| Parameter        | Type    | Description             | Default      |
| ---------------- | ------- | ----------------------- | ------------ |
| `token`          | string  | JWT token               | required     |
| `sessionId`      | string  | Session ID              | required     |
| `message`        | string  | User question           | required     |
| `inputLanguage`  | integer | Input language code     | 23 (English) |
| `outputLanguage` | integer | Response language code  | 23 (English) |
| `context`        | boolean | Use document context    | true         |
| `hasCsvOrXlsx`   | boolean | Session has data files  | false        |
| `mode`           | string  | Processing mode         | "default"    |
| `filenames`      | array   | Specific files to query | []           |

#### Processing Modes

- **`default`**: Fast standard processing
- **`creative`**: Advanced multi-step reasoning (slower, more comprehensive)

#### Response

```json
{
  "answer": "The research identifies three key findings: 1) Machine learning models show 95% accuracy...",
  "fileName": "research_paper.pdf",
  "pageNo": 15,
  "sources": [
    {
      "fileName": "research_paper.pdf",
      "pageNo": 15
    }
  ],
  "questions": [
    "What methodology was used in this research?",
    "What are the limitations of this study?",
    "How do these findings compare to previous work?"
  ]
}
```

#### Creative Mode Response

Creative mode includes additional reasoning metadata:

```json
{
  "answer": "Based on comprehensive analysis...",
  "creative_reasoning": {
    "strategy_used": "Multi-step analysis combining document sections",
    "query_breakdown": [
      "What are the main research questions?",
      "What methods were used?",
      "What results were obtained?"
    ],
    "reasoning_steps": [
      "Step 1: Analyzed methodology section",
      "Step 2: Extracted key results",
      "Step 3: Synthesized findings"
    ],
    "confidence_level": "high"
  }
}
```

---

### Trial Query

Query documents in free trial mode.

**Endpoint**: `POST /trialAsk`  
**Content-Type**: `application/json`  
**Rate Limit**: 100 per minute

#### Request Body

```json
{
  "fingerprint": "browser-fingerprint",
  "message": "Summarize the document",
  "inputLanguage": 23,
  "outputLanguage": 23,
  "context": true
}
```

#### Response

Same format as `/ask` but without creative mode.

---

### Streaming Queries

Get real-time responses for creative mode queries.

**Endpoint**: `GET /ask-stream`  
**Authentication**: Required

#### Query Parameters

Same as `/ask` but passed as URL parameters.

#### Response

Server-sent events (SSE) stream:

```
data: {"type": "strategy_start", "content": "Analyzing your question..."}

data: {"type": "search_start", "content": {"index": 1, "query": "methodology"}}

data: {"type": "answer_chunk", "content": "Based on the analysis..."}

data: {"type": "complete", "content": {"processing_time": 25.4}}
```

---

## 🧠 Knowledge Graphs

### Create Knowledge Graph

Generate visual representation of document relationships.

**Endpoint**: `POST /create_graph`  
**Authentication**: Required

#### Request

```
token: JWT token
sessionId: Session ID
file_0: First filename
file_1: Second filename (optional)
isDense: "true" for dense graph, "false" for high-level
```

#### Response

```json
{
  "message": "Knowledge graph creation started",
  "status": "processing"
}
```

---

### Check Graph Status

Monitor knowledge graph creation progress.

**Endpoint**: `POST /check_graph_status`

#### Response

```json
{
  "status": "completed",
  "message": "Knowledge graph created successfully"
}
```

---

### Fetch Graph Data

Retrieve generated knowledge graph.

**Endpoint**: `GET /fetch_graph`

#### Query Parameters

```
sessionId: Session identifier
token: JWT token
```

#### Response

```json
{
  "nodes": [
    {
      "id": "session_123_AI",
      "name": "Artificial Intelligence",
      "labels": ["Technology"],
      "color": "#FF6B6B"
    }
  ],
  "relationships": [
    {
      "source": "session_123_AI",
      "target": "session_123_ML",
      "type": "INCLUDES"
    }
  ]
}
```

---

## 💳 Account Management

### Update Payment Status

Upgrade user subscription plan.

**Endpoint**: `POST /updatepayment`  
**Content-Type**: `application/json`

#### Request Body

```json
{
  "email": "user@example.com",
  "paymentPlan": 1
}
```

#### Payment Plans

| Plan | Duration | Description            |
| ---- | -------- | ---------------------- |
| 1    | 30 days  | Monthly subscription   |
| 2    | 90 days  | Quarterly subscription |
| 3    | 365 days | Annual subscription    |

#### Response

```json
{
  "message": "Account upgraded successfully!"
}
```

---

### Check Payment Status

Verify subscription status and remaining time.

**Endpoint**: `POST /check-payment-status`

#### Request Body

```json
{
  "email": "user@example.com"
}
```

#### Response

```json
{
  "status": "paid",
  "remaining_days": 25
}
```

---

## 🌐 Multi-language Support

### Supported Languages

| Code | Language | Code | Language  |
| ---- | -------- | ---- | --------- |
| 1    | Hindi    | 13   | Maithili  |
| 2    | Gom      | 14   | Punjabi   |
| 3    | Kannada  | 15   | Malayalam |
| 4    | Dogri    | 16   | Manipuri  |
| 5    | Bodo     | 17   | Telugu    |
| 6    | Urdu     | 18   | Sanskrit  |
| 7    | Tamil    | 19   | Nepali    |
| 8    | Kashmiri | 20   | Santali   |
| 9    | Assamese | 21   | Gujarati  |
| 10   | Bengali  | 22   | Odia      |
| 11   | Marathi  | 23   | English   |
| 12   | Sindhi   |      |           |

### Translation Service

**Endpoint**: `POST /scaler/translate`

#### Request Body

```json
{
  "source_language": 23,
  "content": "Hello world",
  "target_language": 1
}
```

#### Response

```json
{
  "status_code": 200,
  "message": "Translation successful",
  "translated_content": "नमस्ते दुनिया"
}
```

---

## 🔧 Utility Endpoints

### Health Check

Verify API status.

**Endpoint**: `GET /healthcheck`

#### Response

```json
{
  "message": "app is up and running"
}
```

---

### Demo Mode

Query pre-loaded public transport data.

**Endpoint**: `POST /demo`  
**Rate Limit**: 10 per minute

#### Request Body

```json
{
  "message": "How do I get to the airport?"
}
```

#### Response

```json
{
  "answer": "To reach the airport, you can take Metro Line 3..."
}
```

---

## 📱 WhatsApp Integration

### Webhook Endpoint

Handle WhatsApp Business API events.

**Endpoint**: `GET|POST /api/webhook`

- **GET**: Webhook verification
- **POST**: Message processing

This endpoint automatically processes WhatsApp messages and responds using the demo knowledge base.

---

## ❌ Error Handling

### HTTP Status Codes

| Code | Meaning               | Description                       |
| ---- | --------------------- | --------------------------------- |
| 200  | Success               | Request completed successfully    |
| 400  | Bad Request           | Invalid request parameters        |
| 401  | Unauthorized          | Missing or invalid authentication |
| 403  | Forbidden             | Access denied                     |
| 404  | Not Found             | Resource not found                |
| 429  | Too Many Requests     | Rate limit exceeded               |
| 500  | Internal Server Error | Server error                      |

### Error Response Format

```json
{
  "status": "error",
  "message": "Description of the error",
  "error_code": "SPECIFIC_ERROR_CODE"
}
```

### Common Error Messages

| Message                         | Cause                   | Solution                      |
| ------------------------------- | ----------------------- | ----------------------------- |
| "Token is missing!"             | No authentication token | Include token in request      |
| "Token has expired!"            | Expired JWT token       | Refresh authentication        |
| "No data was extracted!"        | File processing failed  | Check file format and content |
| "Free Trial limit is exhausted" | Trial usage exceeded    | Upgrade to paid account       |

---

## 🚀 Example Workflows

### Complete Document Analysis Workflow

```bash
# 1. Upload documents
curl -X POST https://qdocbackend.carnotresearch.com/upload \
  -F "token=$JWT_TOKEN" \
  -F "sessionId=analysis_001" \
  -F "files=@research_paper.pdf" \
  -F "files=@dataset.csv"

# 2. Basic query
curl -X POST https://qdocbackend.carnotresearch.com/ask \
  -H "Content-Type: application/json" \
  -d '{
    "token": "'$JWT_TOKEN'",
    "sessionId": "analysis_001",
    "message": "What are the main findings?",
    "context": true,
    "hasCsvOrXlsx": true
  }'

# 3. Advanced creative mode query
curl -X POST https://qdocbackend.carnotresearch.com/ask \
  -H "Content-Type: application/json" \
  -d '{
    "token": "'$JWT_TOKEN'",
    "sessionId": "analysis_001",
    "message": "Compare the theoretical framework with the experimental results",
    "mode": "creative",
    "context": true
  }'

# 4. Generate knowledge graph
curl -X POST https://qdocbackend.carnotresearch.com/create_graph \
  -F "token=$JWT_TOKEN" \
  -F "sessionId=analysis_001" \
  -F "file_0=research_paper.pdf" \
  -F "isDense=false"
```

### Free Trial Workflow

```bash
# 1. Upload document (trial)
curl -X POST https://qdocbackend.carnotresearch.com/freeTrial \
  -F "fingerprint=unique_browser_id" \
  -F "files=@sample_document.pdf"

# 2. Query document (trial)
curl -X POST https://qdocbackend.carnotresearch.com/trialAsk \
  -H "Content-Type: application/json" \
  -d '{
    "fingerprint": "unique_browser_id",
    "message": "Summarize this document in 3 bullet points"
  }'
```

---

## 🔗 SDKs and Libraries

### Python SDK Example

```python
import requests

class IcarKnoClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.token = token

    def upload_document(self, session_id, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/upload",
                data={"token": self.token, "sessionId": session_id},
                files={"files": f}
            )
        return response.json()

    def query(self, session_id, question, mode="default"):
        response = requests.post(
            f"{self.base_url}/ask",
            json={
                "token": self.token,
                "sessionId": session_id,
                "message": question,
                "mode": mode,
                "context": True
            }
        )
        return response.json()

# Usage
client = IcarKnoClient("https://qdocbackend.carnotresearch.com", "your-token")
client.upload_document("session123", "document.pdf")
result = client.query("session123", "What are the key points?", mode="creative")
```

---

## 📊 Performance & Limits

### Processing Times

- **Document Upload**: 2-10 seconds per MB
- **Standard Query**: 1-3 seconds
- **Creative Mode**: 10-60 seconds
- **Knowledge Graph**: 30-120 seconds

### File Limits

- **Max File Size**: 50MB
- **Max Files per Upload**: 10
- **Supported Formats**: PDF, DOCX, TXT, CSV, XLSX, PPTX

### API Limits

- **Free Trial**: 10 queries total
- **Paid Plans**: Based on subscription tier
- **Rate Limiting**: Implemented per endpoint

---

_For additional support or questions, contact: support@carnotresearch.com_
