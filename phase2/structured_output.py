from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_structured_output(user_message, json_schema):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": f"""You are a data extraction expert.
Always respond with valid JSON only.
No extra text, no markdown, no code blocks.
Follow this exact schema:
{json_schema}"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    raw = response.choices[0].message.content
    print("Raw LLM Response:", raw)
    
    # Clean response and parse JSON
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "")
    return json.loads(raw)


# ---- Example 1: Extract person info ----
schema1 = """{
    "name": "string",
    "experience_years": "number",
    "skills": ["string"],
    "seniority": "junior/mid/senior"
}"""

result1 = get_structured_output(
    "Raj has 18 years of IT experience. He knows Mainframe, Data Engineering, SQL, and Python.",
    schema1
)

print("Example 1 — Person Info:")
print(result1)
print(f"Name: {result1['name']}")
print(f"Skills: {result1['skills']}")
print(f"Seniority: {result1['seniority']}")
print("\n" + "="*50 + "\n")


# ---- Example 2: Analyze sales text ----
schema2 = """{
    "best_product": "string",
    "best_region": "string",
    "total_revenue": "number",
    "recommendation": "string"
}"""

result2 = get_structured_output(
    """Analyze this sales summary:
    Laptop sold 650000, Phone sold 285000, Tablet sold 180000.
    South region had highest sales of 335000.""",
    schema2
)

print("Example 2 — Sales Analysis:")
print(result2)
print(f"Best Product: {result2['best_product']}")
print(f"Recommendation: {result2['recommendation']}")