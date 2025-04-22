# Mental Health Chatbot: User Manual & API Documentation

## Overview
This is a mental health support chatbot built using FastAPI for the backend and React for the frontend. The chatbot detects user sentiment (positive, negative, or neutral) and provides supportive responses, while also tracking mood trends.

## Features
- JWT-based login system
- Real-time WebSocket messaging
- Sentiment classification using a fine-tuned transformer model
- Therapy intent matching based on user mood
- Mood history tracking with visual chart
- Crisis detection support
- Persistent user sessions and chat history

---

## User Manual

### How to Use

#### Login
- Launch the app from your browser (http://localhost:3000 or your deployed URL)
- Enter your **username** and **password**
- Use test credentials for demo:
  - Username: `testuser`
  - Password: `testpassword`

#### Starting a Chat
- Click **+ New Chat** to begin a new session
- Type your message and press Enter or click Send
- The chatbot will respond with contextually appropriate support messages

#### Mood Tracking
- A line chart displays mood sentiment over time
- Sentiment values:
  - `-1`: Negative
  - `0`: Neutral
  - `1`: Positive
- You can view the mood progression with timestamps

#### Chat History
- Each new chat is saved in the sidebar
- Click on a previous session to load its messages
- Option to delete specific chats

#### Logout
- Click the **Logout** button to securely exit the session

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
- Token-based using OAuth2 Password Bearer
- Header:
```
Authorization: Bearer <your_token>
```

---

### `POST /register`
Register a new user.

**Request Body**:
```json
{
  "username": "string",
  "password": "string"
}
```
**Response**:
```json
{
  "message": "User registered successfully"
}
```

---

### `POST /token`
Login and receive access token.

**Form Data**:
- username: `string`
- password: `string`

**Response**:
```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

---

### `POST /chat`
Send a message to the chatbot and receive a response with sentiment.

**Headers**:
```
Authorization: Bearer <token>
```

**Request Body**:
```json
{
  "message": "string"
}
```

**Response**:
```json
{
  "user": "string",
  "user_input": "string",
  "detected_sentiment": "positive | neutral | negative",
  "response": "string"
}
```

---

### `GET /mood-history`
Retrieve historical mood scores for the authenticated user.

**Headers**:
```
Authorization: Bearer <token>
```

**Response**:
```json
{
  "username": "string",
  "moods": [
    {
      "timestamp": "ISO8601 string",
      "mood": "positive | neutral | negative"
    },
    ...
  ]
}
```

---

### WebSocket: `/socket.io/`
Establish real-time two-way chat.

**Auth Header**:
```json
{
  "token": "<JWT access token>"
}
```

**Emit Event**: `chat_message`
```json
{
  "text": "string",
  "timestamp": "ISO8601 string",
  "token": "<token>"
}
```

**Listen for**: `response`
```json
{
  "message": "string"
}
```

---

## Deployment Notes

### Requirements
- Python 3.10+
- Node.js 18+
- Uvicorn for backend
- Nginx (optional) as reverse proxy

### Deployment Options
- **Local**: Run with Uvicorn and React development server
- **Cloud**: Deploy backend to EC2, frontend via S3 + CloudFront or Vercel

### Environment Variables (`.env`)
```
SECRET_KEY=your-secret-key
MODEL_PATH=models/sentiment_model
THERAPY_CSV=data/processed/therapy_processed.csv
```

---

## Contact / Contributions
- This project is designed for educational and mental health support purposes.
- For feedback or contributions, raise a GitHub issue or email the maintainer.

---

*End of documentation.*

