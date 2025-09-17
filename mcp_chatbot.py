import os
import sys
import webbrowser
import argparse
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import requests
import uvicorn

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ Missing GROQ_API_KEY! Please set it in a .env file.")
if not TAVILY_API_KEY:
    raise RuntimeError("❌ Missing TAVILY_API_KEY! Please set it in a .env file.")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Conversation Memory
# -----------------------------
conversation_history = deque(maxlen=10)  # keep last 10 messages

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="MCP Chatbot with Groq + Internet Access")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# -----------------------------
# Tavily Search
# -----------------------------
def tavily_search(query: str) -> str:
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": query, "num_results": 3}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return "No relevant web results found."
        return "\n".join([f"- {res['title']}: {res['url']}" for res in results])
    except Exception as e:
        return f"(Search error: {e})"

# -----------------------------
# Core Chat Function
# -----------------------------
def chatbot_reply(user_msg: str) -> str:
    # Add user message to memory
    conversation_history.append({"role": "user", "content": user_msg})

    # Handle search
    if any(word in user_msg.lower() for word in ["search", "google", "find", "latest"]):
        search_results = tavily_search(user_msg)
        system_prompt = (
            "You are a helpful assistant. Use the following search results to answer.\n"
            f"{search_results}"
        )
    else:
        system_prompt = "You are a helpful assistant."

    # Build messages with memory
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(list(conversation_history))

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()
    # Save assistant reply to memory
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        reply = chatbot_reply(req.message.strip())
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# CLI Mode
# -----------------------------
def run_cli():
    print("🤖 MCP Chatbot (CLI mode with memory). Type 'exit' to quit.\n")
    while True:
        try:
            user_msg = input("You: ").strip()
            if user_msg.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            reply = chatbot_reply(user_msg)
            print(f"Bot: {reply}\n")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Chatbot")
    parser.add_argument("--api", action="store_true", help="Run API server with Swagger UI")
    parser.add_argument("--cli", action="store_true", help="Run chatbot in CLI mode")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        url = "http://127.0.0.1:8000/docs"
        print(f"🚀 Opening {url} ...")
        webbrowser.open(url)
        uvicorn.run("mcp_chatbot:app", host="127.0.0.1", port=8000, reload=True)





