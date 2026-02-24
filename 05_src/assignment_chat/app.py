from weather_chat.main import get_weather_chat_agent
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

# If you have dotenv in your course env:
from dotenv import load_dotenv
load_dotenv(".secrets")
load_dotenv(".env")

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
    title="Weather-Sage ☁️🧠",
    description="Ask for forecasts by city name. Example: “Forecast for Toronto for 3 days.”",
)

if __name__ == "__main__":
    chat.launch()