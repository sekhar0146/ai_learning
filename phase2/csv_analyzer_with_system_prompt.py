from groq import Groq
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_csv(filepath, question):
    # Step 1: Read CSV
    df = pd.read_csv(filepath)
    
    # Step 2: Convert to text
    csv_text = df.to_string(index=False)
    
    # Step 3: Send to LLM with proper system prompt
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": """You are an expert data analyst. 
When answering questions:
- Always use exact numbers from the data
- Show your calculations clearly
- Be concise and direct
- End with a one line conclusion"""
            },
            {
                "role": "user",
                "content": f"Here is the data:\n{csv_text}\n\nQuestion: {question}"
            }
        ]
    )
    
    return response.choices[0].message.content

# Ask multiple questions about the same CSV
questions = [
    "Which product has the highest total sales?",
    "Which region is performing best?",
    "What is the average units sold per day?",
    "Give me a summary of the data in 3 bullet points."
]

## 
filepath = "phase2/sales_data.csv"

for question in questions:
    print(f"Q: {question}")
    print(f"A: {analyze_csv(filepath, question)}")
    print("\n" + "="*50 + "\n")