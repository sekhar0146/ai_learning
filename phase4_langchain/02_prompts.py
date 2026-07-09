from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def basic_prompt_template(llm):
    """
    Simple prompt template with one variable.
    {topic} gets replaced with actual value when called.
    """
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in simple English in 2 sentences."
    )

    # Format prompt with actual value
    formatted = prompt.format(topic="RAG in AI")
    print("Formatted prompt:")
    print(formatted)
    print()

    # Send to LLM
    chain = prompt | llm  # | means "pipe" — connect prompt to llm
    response = chain.invoke({"topic": "RAG in AI"})
    print("Response:")
    print(response.content)
    print()


def system_user_prompt(llm):
    """
    Real world pattern — separate system and user prompts.
    System sets behavior, user sends the actual question.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. Always respond in {language}."),
        ("human", "{question}")
    ])

    chain = prompt | llm

    # Same prompt, different values
    print("Response 1 — Data Engineer in English:")
    response1 = chain.invoke({
        "role": "senior data engineer",
        "language": "English",
        "question": "What is Apache Kafka?"
    })
    print(response1.content)
    print()

    print("Response 2 — HR Assistant:")
    response2 = chain.invoke({
        "role": "HR assistant",
        "language": "English",
        "question": "What is the maternity leave policy?"
    })
    print(response2.content)
    print()


def dynamic_few_shot_prompt(llm):
    """
    Few shot prompting — give examples inside the prompt.
    Model learns the pattern from examples.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You classify HR questions into categories.
Categories: LEAVE, SALARY, WORK_FROM_HOME, PERFORMANCE, OTHER

Examples:
Question: How many sick days do I get? → LEAVE
Question: When is appraisal? → PERFORMANCE
Question: Can I work from home? → WORK_FROM_HOME

Now classify the following question. Reply with category only."""),
        ("human", "{question}")
    ])

    chain = prompt | llm

    questions = [
        "What is the notice period?",
        "How is my increment calculated?",
        "Can I take 2 days off next week?",
    ]

    print("Question Classification:")
    for q in questions:
        response = chain.invoke({"question": q})
        print(f"  Q: {q}")
        print(f"  Category: {response.content}")
        print()


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    print("=" * 50)
    print("1. Basic Prompt Template")
    print("=" * 50)
    basic_prompt_template(llm)

    print("=" * 50)
    print("2. System + User Prompt")
    print("=" * 50)
    system_user_prompt(llm)

    print("=" * 50)
    print("3. Few Shot Prompt")
    print("=" * 50)
    dynamic_few_shot_prompt(llm)


if __name__ == "__main__":
    main()