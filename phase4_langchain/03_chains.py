from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


def simple_chain(llm):
    """
    Basic chain: prompt | llm | parser
    StrOutputParser extracts plain text from response object.
    """
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in exactly 2 sentences."
    )
    parser = StrOutputParser()

    # Build chain
    chain = prompt | llm | parser

    # Without parser — returns message object
    # With parser — returns plain string directly
    result = chain.invoke({"topic": "Vector Database"})

    print("Simple Chain Result:")
    print(result)
    #print(type(result))  # should be <class 'str'>
    print()


def sequential_chain(llm):
    """
    Sequential chain — output of one chain feeds into next.
    Real world use: generate content → translate → summarize
    """
    parser = StrOutputParser()

    # Chain 1 — generate explanation
    prompt1 = ChatPromptTemplate.from_template(
        "Explain {topic} in 3 bullet points for a data engineer."
    )
    chain1 = prompt1 | llm | parser
    chain1_result = chain1.invoke({"topic": "Apache Kafka"})
    print("chain1 result:")
    print(chain1_result)
    print("")

    # Chain 2 — simplify the explanation
    prompt2 = ChatPromptTemplate.from_template(
        "Simplify this explanation for a non-technical manager in 2 sentences:\n{explanation}"
    )
    chain2 = prompt2 | llm | parser

    # Connect both chains
    full_chain = chain1 | (lambda x: {"explanation": x}) | chain2

    result = full_chain.invoke({"topic": "Apache Kafka"})

    print("Sequential Chain Result:")
    print("(Technical explanation → simplified for manager)")
    print(result)
    print()


def batch_chain(llm):
    """
    Batch — run same chain on multiple inputs at once.
    Much faster than running one by one.
    """
    prompt = ChatPromptTemplate.from_template(
        "Classify this HR question into one word: LEAVE, SALARY, WFH, RESIGNATION, or OTHER.\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    questions = [
        {"question": "How many sick days do I get?"},
        {"question": "When is my salary credited?"},
        {"question": "Can I work from home on Friday?"},
        {"question": "What is the notice period?"},
    ]

    # Process all questions at once
    results = chain.batch(questions)

    print("Batch Chain Results:")
    for q, r in zip(questions, results):
        print(f"  Q: {q['question']}")
        print(f"  Category: {r.strip()}")
        print()


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    print("=" * 50)
    print("1. Simple Chain")
    print("=" * 50)
    simple_chain(llm)

    print("=" * 50)
    print("2. Sequential Chain")
    print("=" * 50)
    sequential_chain(llm)

    print("=" * 50)
    print("3. Batch Chain")
    print("=" * 50)
    batch_chain(llm)


if __name__ == "__main__":
    main()