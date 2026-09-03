---
title: Company Research Agent
emoji: 🤖
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
tags:
- streamlit
- langchain
- groq
- tavily
- ai-agent
pinned: false
short_description: AI agent for company research and briefings
---

# Company Research Agent 🤖

An autonomous AI agent that independently researches any company and generates a structured analyst briefing — powered by LangChain, Groq, and Tavily Search.

## Live Demo

**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/krishiawasthi/company-research-agent)**

---

## What it does

Enter any company name and the agent will:

1. Independently decide what to search for
2. Execute multiple live web searches using Tavily
3. Reason over the results and decide if more searches are needed
4. Generate a structured analyst briefing covering:
   - What the company does
   - Recent developments
   - Financial highlights
   - Key risks
   - Overall outlook

This is a **ReAct-based agentic system** — the agent reasons and acts in a loop, making its own decisions about how many searches to run and what to look for, without step-by-step human instruction.

---

## How it works

User input (company name)
        ↓
ReAct Agent (LangChain + Groq)
        ↓
Tool calls → Tavily Web Search (live results)
        ↓
Reasoning loop (repeat until sufficient data)
        ↓
Structured analyst briefing output

---

## Tech stack

- **LangChain** — agent framework and ReAct reasoning loop
- **Groq API** — LLM inference (fast, free tier available)
- **Tavily** — real-time web search tool for agents
- **Streamlit** — web interface
- **Python** — core language

---

## Project structure

company-research-agent/
├── app.py                 # Main Streamlit app and agent logic
├── agent.py               # Agent configuration
├── requirements.txt       # Dependencies
└── README.md

---

## Run locally

git clone https://github.com/krishiawasthi/company-research-agent
cd company-research-agent
pip install -r requirements.txt

Create a .env file:
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

Run:
python3 -m streamlit run app.py

---

## Skills demonstrated

- Agentic AI architecture (ReAct pattern)
- LLM tool use and multi-step reasoning
- Prompt engineering for structured outputs
- Real-time web search integration
- Python application deployment on Hugging Face Spaces