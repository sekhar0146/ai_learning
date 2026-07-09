from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


def summarize_history(llm, history, existing_summary=""):
    """
    Summarizes conversation history into a short paragraph.
    This replaces old messages — keeps context without growing forever.
    """
    if not history:
        return existing_summary

    # Convert history to readable text
    history_text = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        else:
            history_text += f"AI: {msg.content}\n"

    prompt = ChatPromptTemplate.from_template("""
You are summarizing a conversation for memory purposes.
Keep all important facts — names, preferences, key information discussed.
Be concise — maximum 3 sentences.

Existing summary: {existing_summary}

New conversation to add:
{history_text}

Updated summary:""")

    chain = prompt | llm | StrOutputParser()

    summary = chain.invoke({
        "existing_summary": existing_summary,
        "history_text": history_text
    })

    return summary


def summary_memory_chat(llm, summarize_every=2):
    """
    Summary Memory — summarizes old messages instead of dropping them.
    Best of both worlds — remembers everything, controls token usage.
    summarize_every=2 means summarize after every 2 exchanges.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful HR assistant.
Use this conversation summary for context: {summary}"""),
        MessagesPlaceholder(variable_name="recent_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    history = []
    summary = ""

    questions = [
        "My name is Raj and I work in data engineering.",
        "I have 18 years of IT experience.",
        "How many sick days do I get?",
        "Can those days be carried forward?",
        "What is my name and what do I do?",  # tests long term memory
    ]

    for i, question in enumerate(questions):
        print(f"Q: {question}")

        # Summarize old history every N exchanges
        if len(history) >= summarize_every * 2:
            old_history = history[:-summarize_every * 2]  # older messages
            recent_history = history[-summarize_every * 2:]  # recent messages

            if old_history:
                summary = summarize_history(llm, old_history, summary)
                print(f"   📝 Summary updated: {summary[:100]}...")
        else:
            recent_history = history

        response = chain.invoke({
            "summary": summary if summary else "No previous context.",
            "recent_history": recent_history,
            "question": question
        })

        print(f"A: {response}")
        print(f"   History: {len(history)} msgs | Summary: {'Yes' if summary else 'No'}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response))


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    print("=" * 50)
    print("Summary Memory — Summarizes Old Context")
    print("=" * 50 + "\n")
    summary_memory_chat(llm, summarize_every=2)


if __name__ == "__main__":
    main()