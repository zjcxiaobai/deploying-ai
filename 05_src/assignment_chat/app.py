from assignment_chat.main import get_weather_chat_agent
from assignment_chat.semantic_store import SemanticStore
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from assignment_chat.service3_web import wiki_web_search
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# gateway key to use openAI
BASE_DIR = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent          # .../05_src/assignment_chat
ROOT_05_SRC = HERE.parent                      # .../05_src
load_dotenv(ROOT_05_SRC / ".secrets", override=True)
load_dotenv(ROOT_05_SRC / ".env", override=True)
# service 2
STORE = SemanticStore(
    persist_dir=ROOT_05_SRC / "assignment_chat" / "chroma_store",
    data_csv=ROOT_05_SRC / "assignment_chat" / "test.csv",
)
STORE.ingest_from_csv()


llm = get_weather_chat_agent()

RESTRICTED = [
    # assignment bans:
    "cat", "cats", "dog", "dogs",
    "horoscope", "zodiac",
    "taylor swift",
]

PROMPT_ATTACK_HINTS = [
    "system prompt", "reveal your prompt", "show me the prompt",
    "ignore previous instructions", "developer message", "jailbreak",
]

def violates_rules(text: str) -> bool:
    t = (text or "").lower()
    if any(k in t for k in RESTRICTED):
        return True
    if any(k in t for k in PROMPT_ATTACK_HINTS):
        return True
    return False

def weather_chat(message: str, history: list[dict]) -> str:
    # Guardrails at the UI boundary (simple + effective)
    if violates_rules(message):
        return (
            "Can’t help with that topic/request. "
            "Ask me about weather instead (e.g., “What’s the forecast for Toronto?”)."
        )
    # add service 2 
    if message.strip().lower().startswith("/search"):
        q = message.split(" ", 1)[1].strip() if " " in message else ""
        if not q:
            return "Usage: /search <your question>"
        hits = STORE.search(q, k=5)
        if not hits:
            return "No results found."

        lines = [f"Top matches for: **{q}**"]
        for i, h in enumerate(hits, 1):
            snippet = h["text"][:200] + ("…" if len(h["text"]) > 200 else "")
            title = h["title"] or h["id"]
            lines.append(f"{i}. **{title}** — {snippet}")
        return "\n".join(lines)
    # add service 3
    if message.strip().lower().startswith("/web"):
        q = message.split(" ", 1)[1].strip() if " " in message else ""
        return wiki_web_search(q, k=3)
    # ###########################
    langchain_messages = []
    n = 0
    for msg in history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
            n += 1

    langchain_messages.append(HumanMessage(content=message))

    state = {"messages": langchain_messages, "llm_calls": n}
    response = llm.invoke(state)

    # last message content
    return response["messages"][-1].content


chat = gr.ChatInterface(
    fn=weather_chat,
    type="messages",
    title="Weather-AI",
    description=(
        "**Service 1 (Weather API)**<br>"
        "• Example: `Forecast for Toronto for 3 days`<br><br>"
        "**Service 2 (Semantic Search)**<br>"
        "• Use: `/search <question>`<br>"
        "• Example: `/search precipitation probability`<br><br>"
        "**Service 3 (Web Search)**<br>"
        "• Use: `/web <term>`<br>"
        "• Example: `/web climate change`<br><br>"
    )
)

if __name__ == "__main__":
    chat.launch()