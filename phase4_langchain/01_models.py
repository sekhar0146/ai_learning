from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


def basic_call(llm):
    """
    Simplest possible LangChain call.
    .invoke() takes a string and returns a response object.
    """
    response = llm.invoke("What is a data pipeline in one sentence?")
    print("Basic call:")
    print(response.content)
    print()


def call_with_messages(llm):
    """
    Real world way — using system + user messages.
    Same concept as Phase 2 but cleaner syntax.
    """
    messages = [
        SystemMessage(content="You are a data engineering expert. Be concise."),
        HumanMessage(content="What are the 3 stages of ETL?")
    ]
    response = llm.invoke(messages)
    print("Call with messages:")
    print(response.content)
    print()


def streaming_call(llm):
    """
    Streaming — response comes word by word.
    Used in chatbots so user sees response as it generates.
    """
    print("Streaming call:")
    for chunk in llm.stream("Tell me 3 benefits of data engineering in one line each."):
        print(chunk.content, end="", flush=True)
    print("\n")


def main():
    # Initialize model — reads GROQ_API_KEY from .env automatically
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    print("=" * 50)
    basic_call(llm)
    print("=" * 50)
    call_with_messages(llm)
    print("=" * 50)
    streaming_call(llm)


if __name__ == "__main__":
    main()