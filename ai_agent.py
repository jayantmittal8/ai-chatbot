
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup API Keys for Groq, OpenAI and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

#Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]

# Step4: Interactive Chat Loop
if __name__ == "__main__":
    print("🤖 AI Chatbot is ready! Type 'exit' to quit.\n")
    provider = input("Choose provider (OpenAI/Groq): ").strip() or "OpenAI"
    llm_id = "gpt-4o-mini" if provider == "OpenAI" else "llama-3.3-70b-versatile"

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        response = get_response_from_ai_agent(
            llm_id=llm_id,
            query=query,
            allow_search=True,   # set to False if you don’t want Tavily search
            system_prompt=system_prompt,
            provider=provider
        )
        print("AI:", response, "\n")
