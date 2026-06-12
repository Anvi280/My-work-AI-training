import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from date_time_tool import get_current_datetime

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=your_key")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
tools = [get_current_datetime]
agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant. If the user asks for the current date or time, use the get_current_datetime tool.")


def format_response(response):
    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
            if isinstance(last_message, dict):
                return last_message.get("content") or str(last_message)
        return str(response)
    return getattr(response, "output_text", None) or getattr(response, "output", None) or getattr(response, "text", None) or str(response)


def chat():
    print("=" * 60)
    print("LangChain Chatbot with real-time date/time tool")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "end"}:
            print("Goodbye!")
            break

        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print("Bot:", format_response(response))
        print()


if __name__ == "__main__":
    chat()
