"""
Re-implemented chatbot using LangChain with Memory
Based on 05-simple-application with patterns from 12-chains-memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)

# Create prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions concisely."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain
chain = prompt | llm

# Store conversation histories for different sessions
store = {}

def get_session_history(session_id: str):
    """Get or create session history"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create conversation with memory
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat():
    """Main chat function"""
    print("=" * 60)
    print("🤖 LangChain Chatbot with Memory")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'bye' to end conversation")
    print()
    
    session_id = "user_1"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Exit conditions
            if user_input.lower() in ["quit", "exit", "bye", "end"]:
                print("Bot: Goodbye! Thanks for chatting.")
                break
            
            # Empty input
            if not user_input:
                continue
            
            # Get response from LLM with memory
            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            print(f"Bot: {response.content}\n")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    chat()