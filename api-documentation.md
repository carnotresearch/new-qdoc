# icarKno API Documentation

## Overview

icarKno (I carry Knowledge) is an intelligent document analysis API that allows users to upload documents, process them, and query their content through natural language. This documentation provides details about all available endpoints, request/response formats, and usage examples.

## Base URL

```
https://qdocbackend.carnotresearch.com
```

## Authentication

Most endpoints require JWT-based authentication. Include the token in request forms or JSON body where specified.

## Rate Limits

- Free trial users: 10 queries per session
- Paid users: Unlimited queries based on subscription level

---

## Document Management Endpoints

### Upload Documents (New Session)

Upload documents to create a new session.

**URL**: `/upload`  
**Method**: `POST`  
**Content-Type**: `multipart/form-data`

**Request Parameters**:
- `token` (string, required): JWT authentication token
- `sessionId` (string, required): Unique session identifier
- `files` (file array, required): Documents to upload (PDF, DOCX, TXT, CSV, or XLSX)

**Response**:
```json
{
  "status": "ok",
  "message": "Files successfully uploaded."
}
```

**Status Codes**:
- `200`: Success
- `401`: Authentication failed
- `500`: Error processing files

---

### Add Documents to Existing Session

Add additional documents to an existing session.

**URL**: `/add-upload`  
**Method**: `POST`  
**Content-Type**: `multipart/form-data`

**Request Parameters**:
- `token` (string, required): JWT authentication token
- `sessionId` (string, required): Existing session identifier
- `files` (file array, required): Additional documents to upload

**Response**:
```json
{
  "status": "ok",
  "message": "Files successfully uploaded."
}
```

**Status Codes**:
- `200`: Success
- `401`: Authentication failed
- `500`: Error processing files

---

### Free Trial Upload

Upload documents for a free trial session.

**URL**: `/freeTrial`  
**Method**: `POST`  
**Content-Type**: `multipart/form-data`  
**Rate Limit**: 10 per minute

**Request Parameters**:
- `fingerprint` (string, required): Browser fingerprint for session tracking
- `files` (file array, required): Documents to upload

**Response**:
```json
{
  "status": "ok",
  "message": "Files uploaded successfully"
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid fingerprint
- `500`: Error processing files

---

## Querying Endpoints

### Ask Question (Authenticated)

Query documents with natural language for authenticated users.

**URL**: `/ask`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "token": "your-jwt-token",
  "sessionId": "session-identifier",
  "message": "What is the main conclusion of the document?",
  "inputLanguage": 23,
  "outputLanguage": 23,
  "context": true,
  "hasCsvOrXlsx": false,
  "mode": 1
}
```

**Parameters**:
- `token` (string, required): JWT authentication token
- `sessionId` (string, required): Session identifier
- `message` (string, required): Natural language query
- `inputLanguage` (integer, required): Language code of input (23 for English)
- `outputLanguage` (integer, required): Language code for response
- `context` (boolean, required): Whether to use document context
- `hasCsvOrXlsx` (boolean, optional): Whether session includes tabular data
- `mode` (integer, optional): 1 for contextual mode, 2 for creative mode

**Response**:
```json
{
  "answer": "The main conclusion of the document states that..."
}
```

**Status Codes**:
- `200`: Success
- `401`: Authentication failed
- `500`: Error generating response

---

### Trial Ask Question

Query documents for free trial users.

**URL**: `/trialAsk`  
**Method**: `POST`  
**Content-Type**: `application/json`  
**Rate Limit**: 100 per minute

**Request Body**:
```json
{
  "fingerprint": "browser-fingerprint",
  "message": "Summarize the document",
  "inputLanguage": 23,
  "outputLanguage": 23,
  "context": true,
  "hasCsvOrXlsx": false,
  "mode": 1
}
```

**Response**:
```json
{
  "answer": "The document covers..."
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid fingerprint
- `500`: Error generating response

---

### Demo Query

Access pre-loaded documents for demonstration.

**URL**: `/demo`  
**Method**: `POST`  
**Content-Type**: `application/json`  
**Rate Limit**: 10 per minute

**Request Body**:
```json
{
  "message": "What information does this document contain?"
}
```

**Response**:
```json
{
  "answer": "The document contains information about..."
}
```

**Status Codes**:
- `200`: Success
- `500`: Error generating response

---

## Account Management Endpoints

### Update Payment Status

Update a user's subscription plan.

**URL**: `/updatepayment`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "email": "user@example.com",
  "paymentPlan": 1
}
```

**Parameters**:
- `email` (string, required): User's email address
- `paymentPlan` (integer, required): Subscription level (1: 30 days, 2: 90 days, 3: 365 days)

**Response**:
```json
{
  "message": "Account upgraded successfully!"
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid plan or email
- `500`: Database error

---

### Check Payment Status

Check a user's subscription status.

**URL**: `/check-payment-status`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "email": "user@example.com"
}
```

**Response**:
```json
{
  "status": "paid",
  "remaining_days": 25
}
```

**Status Codes**:
- `200`: Success
- `500`: Error checking status

---

## Utility Endpoints

### Health Check

Check if the API server is running.

**URL**: `/healthcheck`  
**Method**: `GET`

**Response**:
```json
{
  "message": "app is up and running"
}
```

---

## Translation API (FastAPI)

### List Supported Languages

Get a list of supported languages.

**URL**: `/`  
**Method**: `GET`

**Response**:
```json
{
  "1": "Hindi",
  "2": "Gom",
  ...
  "23": "English"
}
```

---

### Translate Text

Translate text between supported languages.

**URL**: `/scaler/translate`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:
```json
{
  "source_language": 23,
  "content": "Hello world",
  "target_language": 1
}
```

**Parameters**:
- `source_language` (integer, required): Source language code
- `content` (string, required): Text to translate
- `target_language` (integer, required): Target language code

**Response**:
```json
{
  "status_code": 200,
  "message": "Translation successful",
  "translated_content": "नमस्ते दुनिया"
}
```

**Status Codes**:
- `200`: Success
- Various error codes for translation failures

---

## Language Codes

| Code | Language  |
|------|-----------|
| 1    | Hindi     |
| 2    | Gom       |
| 3    | Kannada   |
| 4    | Dogri     |
| 5    | Bodo      |
| 6    | Urdu      |
| 7    | Tamil     |
| 8    | Kashmiri  |
| 9    | Assamese  |
| 10   | Bengali   |
| 11   | Marathi   |
| 12   | Sindhi    |
| 13   | Maithili  |
| 14   | Punjabi   |
| 15   | Malayalam |
| 16   | Manipuri  |
| 17   | Telugu    |
| 18   | Sanskrit  |
| 19   | Nepali    |
| 20   | Santali   |
| 21   | Gujarati  |
| 22   | Odia      |
| 23   | English   |

## Webhook Integration

### WhatsApp Webhook

Endpoint for WhatsApp Business API integration.

**URL**: `/api/webhook`  
**Method**: `POST` and `GET`

This endpoint handles WhatsApp message events and verification challenges. Implementation details are available upon request.

---

## Error Codes and Messages

| Status Code | Message                         | Description                               |
|-------------|--------------------------------|-------------------------------------------|
| 400         | "Token is missing!"            | Authentication token not provided          |
| 401         | "Token has expired!"           | Expired authentication token              |
| 401         | "Token is invalid!"            | Invalid authentication token              |
| 500         | "No data was extracted!"       | Failed to extract content from files      |
| 500         | "Error creating vector index"  | Error in document processing              |
| 500         | "Error generating response from LLM" | Error in generating AI response      |
