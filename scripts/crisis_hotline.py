CRISIS_KEYWORDS = [
    "suicidal", "want to die", "end my life", "kill myself", "can't do this", "hurt myself"
]

CRISIS_HOTLINES = {
    "USA": "National Suicide & Crisis Lifeline: Call 988 or 1-800-273-8255",
    "UK": "Samaritans: Call 116 123",
    "Canada": "Talk Suicide Canada: Call 1-833-456-4566",
    "India": "iCall: Call +91-22-25521111",
    "Global": "Visit https://www.befrienders.org for local help"
}

def detect_crisis(message: str) -> bool:
    """Detect if message contains crisis keywords."""
    message = message.lower()
    return any(keyword in message for keyword in CRISIS_KEYWORDS)

def get_crisis_response() -> str:
    """Return a response with hotline information."""
    response = "I'm really sorry to hear you're feeling this way. Please consider reaching out to a crisis hotline:\n\n"
    for country, info in CRISIS_HOTLINES.items():
        response += f"🌍 {country}: {info}\n"
    response += "\nYou are not alone. Please seek help immediately. 🙏"
    return response
