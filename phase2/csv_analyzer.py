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
    
    # Step 3: Build prompt
    prompt = f"""You are a data analyst expert.
    
Here is the data:
{csv_text}

Answer this question about the data:
{question}

Be specific, use numbers from the data, and be concise."""

    # Step 4: Send to LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
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

filepath = "phase2/sales_data.csv"

for question in questions:
    print(f"Q: {question}")
    print(f"A: {analyze_csv(filepath, question)}")
    print("\n" + "="*50 + "\n")

# Ask custom questions interactively
print("\n🤖 Ask anything about the data (type 'exit' to quit):\n")
while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    print(analyze_csv(filepath, question))
    print()