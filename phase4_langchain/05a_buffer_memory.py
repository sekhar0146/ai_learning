from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


def buffer_memory_chat(llm):
    """
    Buffer Memory — stores FULL conversation history.
    Sends ALL messages to model every time.
    Good for: short conversations.
    Risk: grows forever, expensive for long conversations.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    history = []

    questions = [
        "My name is Raj and I work in data engineering.",
        "How many sick days do I get?",
        "Can those days be carried forward?",
        "What is my name?"
    ]

    for question in questions:
        print(f"Q: {question}")

        response = chain.invoke({
            "history": history,
            "question": question
        })

        print(f"A: {response}")
        print(f"   History size: {len(history)} messages\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response))


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    print("=" * 50)
    print("Buffer Memory — Full History")
    print("=" * 50 + "\n")
    buffer_memory_chat(llm)


if __name__ == "__main__":
    main()