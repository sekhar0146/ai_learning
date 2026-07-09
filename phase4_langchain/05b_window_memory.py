from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


def window_memory_chat(llm, window_size=2):
    """
    Window Memory — stores only last N exchanges.
    Sends only recent messages to model — controls cost.
    Good for: long conversations where recent context matters most.
    Risk: forgets old context beyond window.
    window_size=2 means keep last 2 exchanges (4 messages).
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
        "What is my name?",  # should forget — outside window
    ]

    for question in questions:
        # Only send last window_size exchanges
        windowed_history = history[-(window_size * 2):]

        print(f"Q: {question}")
        response = chain.invoke({
            "history": windowed_history,
            "question": question
        })

        print(f"A: {response}")
        print(f"   Full history: {len(history)} messages | Sent to model: {len(windowed_history)} messages\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response))


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    print("=" * 50)
    print("Window Memory — Last 2 Exchanges Only")
    print("=" * 50 + "\n")
    window_memory_chat(llm, window_size=2)


if __name__ == "__main__":
    main()