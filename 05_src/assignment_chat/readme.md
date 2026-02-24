# Weather-Sage Chat (Assignment 2)

This is a simple conversational AI system with a Gradio chat UI and a LangGraph agent.

## Personality
Weather-Sage is a nerdy but friendly weather mentor.

## Services
### Service 1: API Calls (Open-Meteo)
- Tool: `get_weather_forecast(location, days)`
- Uses Open-Meteo geocoding + forecast endpoints (no API key).
- Output is transformed into a human-readable forecast summary (no raw JSON).

### Service 2 / Service 3
Add your semantic search service (Chroma persistent) and a third service (function calling/web search/etc.) here.

## Guardrails
- Blocks attempts to reveal/modify system prompt.
- Refuses restricted topics: cats/dogs, horoscopes/zodiac, Taylor Swift.

## Run
```bash
python app.py