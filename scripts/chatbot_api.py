from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import random
import json
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import socketio
from scripts.crisis_hotline import detect_crisis, get_crisis_response
from dotenv import load_dotenv
import logging
from flask import Flask, request, jsonify
from scripts.encryption import encrypt_message, decrypt_message

# --- Load environment variables ---
load_dotenv()

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Setup ---
app = FastAPI(title="Mental Health Chatbot API with Authentication", version="1.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app = Flask(__name__)

# --- Security and JWT Setup ---
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Dummy User Database ---
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": pwd_context.hash("testpassword"),
    }
}

# --- Load Sentiment Model & Tokenizer ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# --- Load Therapy Intents Dataset ---
THERAPY_CSV = os.getenv("THERAPY_CSV", "data/processed/therapy_processed.csv")
therapy_df = pd.read_csv(THERAPY_CSV)

# --- Load Label Mapping ---
with open(f"{MODEL_PATH}/label_mapping.json", "r") as f:
    label_mapping = json.load(f)
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# --- User Mood Persistence ---
MOOD_FILE = "user_moods.json"

def load_user_moods():
    if not os.path.exists(MOOD_FILE):
        return {}
    try:
        with open(MOOD_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}

def save_user_moods(moods):
    with open(MOOD_FILE, "w") as f:
        json.dump(moods, f, indent=4)

user_moods = load_user_moods()

# --- Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserRegister(BaseModel):
    username: str
    password: str

class UserMessage(BaseModel):
    message: str

class User(BaseModel):
    username: str

# --- Auth & JWT Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if user and verify_password(password, user['hashed_password']):
        return User(username=username)
    return None

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Dependency ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise credentials_exception
        return User(username=username)
    except JWTError:
        raise credentials_exception

# --- Auth Routes ---
@app.post("/register")
async def register_user(user: UserRegister):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists.")
    fake_users_db[user.username] = {
        "username": user.username,
        "hashed_password": get_password_hash(user.password),
    }
    return {"message": "User registered successfully"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Sentiment Classification ---
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return inverse_label_mapping[predicted_class_id]

# --- Response Generation ---
SENTIMENT_TO_TAGS = {
    "negative": ["self-esteem-A", "self-esteem-B", "relationship-b", "relationship-a",
                 "angermanagement-a", "angermanagement-b", "domesticviolence",
                 "griefandloss", "substanceabuse-a", "substanceabuse-b", "family-conflict"],
    "neutral": ["family-conflict", "relationship-b"],
    "positive": ["greetings", "goodbye"]
}

def generate_response(user_input):
    if detect_crisis(user_input):
        return "crisis", get_crisis_response()

    sentiment = classify_sentiment(user_input)
    relevant_tags = SENTIMENT_TO_TAGS.get(sentiment, [])
    filtered_responses = therapy_df[therapy_df['tag'].isin(relevant_tags)]

    if filtered_responses.empty:
        return sentiment, "I'm here to listen. Can you tell me more about what you're going through?"

    selected_response = random.choice(filtered_responses['response'].tolist()).split('|')[0]
    return sentiment, selected_response

# --- Chat Route ---
@app.post("/chat")
async def chat_with_bot(user_message: UserMessage, current_user: User = Depends(get_current_user)):
    user_input = user_message.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message is not allowed.")

    sentiment, response = generate_response(user_input)

    if current_user.username not in user_moods:
        user_moods[current_user.username] = []

    user_moods[current_user.username].append({
        "timestamp": datetime.utcnow().isoformat(),
        "mood": sentiment
    })

    save_user_moods(user_moods)

    return {
        "user": current_user.username,
        "user_input": user_input,
        "detected_sentiment": sentiment,
        "response": response
    }

# --- Mood History ---
@app.get("/mood-history")
async def get_mood_history(current_user: User = Depends(get_current_user)):
    user_moods = load_user_moods()
    mood_history = user_moods.get(current_user.username, [])
    return {"username": current_user.username, "moods": mood_history}

@app.route("/api/message", methods=["POST"])
def message():
    encrypted = request.json.get("message")
    user_msg = decrypt_message(encrypted)

    print("Decrypted message:", user_msg)

    # Your existing logic
    bot_reply = f"You said: {user_msg}"  # Replace with actual response generator

    encrypted_reply = encrypt_message(bot_reply)
    return jsonify({"message": encrypted_reply})

# --- WebSocket Support ---
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    auth_header='Authorization'
)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

@sio.event
async def connect(sid, environ, auth):
    logger.info(f"Client connected: {sid}")
    await sio.emit("response", {"message": "Welcome to the chatbot!"}, room=sid)

@sio.event
async def chat_message(sid, data):
    logger.info(f"Message from {sid}: {data}")
    user_input = data.get("text", "").strip()
    token = data.get("token", "")
    timestamp = data.get("timestamp", datetime.utcnow().isoformat())

    if not user_input:
        await sio.emit("response", {"message": "Please enter a message."}, room=sid)
        return

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username or username not in fake_users_db:
            raise Exception("Invalid user")
    except Exception as e:
        await sio.emit("response", {"message": f"Auth error: {str(e)}"}, room=sid)
        return

    sentiment, response = generate_response(user_input)

    if username not in user_moods:
        user_moods[username] = []

    user_moods[username].append({
        "timestamp": timestamp,
        "mood": sentiment
    })

    save_user_moods(user_moods)

    await sio.emit("response", {"message": f"Sentiment Detected: {sentiment}"}, room=sid)
    await sio.emit("response", {"message": response}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

# --- Health Check ---
@app.get("/")
async def root():
    return {"message": "Mental Health Chatbot API with Authentication is running."}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, reload=True)
