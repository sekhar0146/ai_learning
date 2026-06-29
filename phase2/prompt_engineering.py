from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(system, user):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    print(response.choices[0].message.content)
    print("\n" + "="*50 + "\n")

# ---- Technique 1: Vague vs Specific ----

# ask(
#     system="You are a helpful assistant.",
#     user="Tell me about data pipelines."
# )

# ask(
#     system="You are a helpful assistant.",
#     user="""Tell me about data pipelines. Cover these points:
#     1. What they are in one sentence
#     2. The 3 main stages
#     3. Two real world examples
#     Keep the total response under 150 words."""
# )

# ask(
#     system="You are a senior data engineer with 15 years experience in big data systems.",
#     user="What are the most common mistakes junior data engineers make?"
# )

# ask(
#     system="You are a helpful assistant.",
#     user="""Classify these pipeline errors as either 'data quality' or 'infrastructure':
    
#     Example 1: "Disk full on worker node" → infrastructure
#     Example 2: "Null values in required field" → data quality
    
#     Now classify these:
#     1. CPU spike on Spark executor
#     2. Duplicate records found
#     3. Network timeout between nodes
#     4. Invalid date format in column"""
# )

ask(
    system="You are a helpful assistant. Always respond in JSON format only. No extra text.",
    user="""Give me 3 popular data pipeline tools with this structure:
    {
        "tools": [
            {"name": "...", "use_case": "...", "difficulty": "easy/medium/hard"}
        ]
    }"""
)

ask(
    system="You are a helpful assistant.",
    user="""A data pipeline runs every hour. 
    It takes 45 minutes to complete.
    It started failing after we added a new transformation that adds 20 minutes.
    
    Think step by step: Will the pipeline cause overlap issues? What should we do?"""
)