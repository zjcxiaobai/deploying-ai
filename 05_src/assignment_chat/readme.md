# Assignment 2 — Weather-AI (Weather-Sage)


## Introduction
I am a Physics student, for this project I chose weather and atmospheric data as the main topic

This project allows users to:

Access real-time and short-term weather forecasts

Explore weather-related concepts through semantic search

Look up scientific background information using web search

## Project structure
05_src/
└── assignment_chat/
    ├── app.py                 # Gradio chat interface and routing
    ├── main.py                # Weather agent + LLM logic
    ├── semantic_store.py      # ChromaDB semantic search
    ├── service3_web.py        # Wikipedia web search
    ├── make_data.py           # Dataset generator
    ├── test.csv               # Semantic search dataset
    └── chroma_store/          # Persistent vector database

This project implements a conversational AI system with a Gradio chat interface and **three services**:
1) **Service 1 (API Calls):** Weather forecast via Open-Meteo (geocoding + forecast)
2) **Service 2 (Semantic Query):** Semantic search over a local CSV dataset using ChromaDB PersistentClient
3) **Service 3 (Web Search):** Simple web search using Wikipedia (MediaWiki API + page summaries)

## Setup

### 1) openAI gateway key

make sure have the openAI gateway key in `05_src/.secrets`

### 2) generate service 2 dataset

`python 05_src/assignment_chat/make_data.py`

This creates : `05_src/assignment_chat/test.csv`

### 3) run the chat app

`PYTHONPATH=05_src python 05_src/assignment_chat/app.py`

open the Gradio link in browser: `http://127.0.0.1:7860`


---

## How to use the app

### Service 1: ask in natural language

Forecast for Toronto for 3 days

### Service 2:

/search precipitation probability

### Service 3:

/web climate change

