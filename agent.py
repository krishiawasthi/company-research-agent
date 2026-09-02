import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq                          # ← Groq, not Gemini
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="web_search",
        func=search.run,
        description="""Use this to search the web for current information.
        Input should be a specific search query string.
        Use this when you need recent news, financial data,
        or any facts about a company or topic."""
    )
]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=12,
    handle_parsing_errors=True
)

print("\n" + "="*50)
print("       COMPANY RESEARCH AGENT")
print("="*50)

company = input("\nEnter a company name to research: ")

task = f"""
Research the company '{company}' and write a short analyst briefing.

Your briefing must include:
1. What the company does (1-2 sentences)
2. Recent news or developments
3. Any financial highlights (revenue, growth, funding)
4. One key risk or challenge they face
5. Overall outlook — growing, stable, or struggling?

Use the web_search tool to find this information.
Search multiple times if needed to get complete information.
Write the final briefing in clear professional language.
"""

print(f"\nResearching {company}... watch the agent think:\n")

result = agent_executor.invoke({"input": task})

print("\n" + "="*50)
print("         FINAL ANALYST BRIEFING")
print("="*50)
print(result["output"])