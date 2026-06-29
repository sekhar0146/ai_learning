from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list holds the entire conversation history
conversation_history = []

def chat(user_message):
    # Step 1: Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    # Step 2: Send full history to LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful data engineering mentor. Remember everything the user tells you."
            }
        ] + conversation_history  # full history sent every time
    )
    
    # Step 3: Extract reply
    reply = response.choices[0].message.content
    
    # Step 4: Add reply to history
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })
    
    return reply

# Simulate a conversation
print(chat("Hi, my name is Raj and I have 18 years of IT experience."))
print("="*50)
print(chat("I am learning AI right now."))
print("="*50)
print(chat("What do you know about me so far?"))
print("="*50)
print(chat("Based on my background, what AI topics should I focus on?"))