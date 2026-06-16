import os
import streamlit as st
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

st.set_page_config(page_title="Company Research Agent", page_icon="🤖", layout="wide")
st.title("🤖 Company Research Agent")
st.write("Enter a company name — the agent searches the web and writes a professional analyst briefing.")

with st.sidebar:
    st.header("⚙️ Settings")
    max_iter = st.slider("Max agent iterations", min_value=5, max_value=15, value=10)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. You enter a company name")
    st.markdown("2. Agent decides what to search")
    st.markdown("3. Searches web multiple times")
    st.markdown("4. Writes a professional briefing")
    st.divider()
    st.markdown("Built with LangChain · Groq · Tavily · Streamlit")

company = st.text_input("Enter a company name:", placeholder="e.g. Apple, Ryanair, Revolut, Bata...")
run_button = st.button("�� Research this company", type="primary")

if run_button and company:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def tavily_search(query):
        query = query.strip().replace('"', '').replace("'", "")
        try:
            response = tavily.search(query=query, max_results=5)
            results = response.get("results", [])
            if not results:
                return "No results found."
            output = ""
            for r in results:
                output += f"Source: {r['title']}\nContent: {r['content'][:400]}\n\n"
            return output
        except Exception as e:
            return f"Search error: {str(e)}"

    tools = [
        Tool(
            name="web_search",
            func=tavily_search,
            description="""Search the web for current information about a company.
            Always use plain text queries with no quotes.
            Good: Bata company overview
            Good: Bata revenue 2024
            Search multiple times to cover different aspects."""
        )
    ]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max_iter,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    task = f"""Research the company {company} and write a professional analyst briefing.

Search for: overview, recent news, financial highlights, key risks.
Use plain text search queries with no quote marks.

Write your final answer in exactly this format:

---

## {company} — Analyst Briefing

### What they do
[2 sentences describing the business]

### Recent developments
- [Point 1]
- [Point 2]
- [Point 3]

### Financial highlights
[2 sentences with key figures]

### Key risk
[1 sentence on biggest challenge]

### Overall outlook
[1 sentence — growing, stable, or under pressure]

---

Only use facts from your searches. Do not invent numbers."""

    st.divider()
    st.subheader(f"Researching {company}...")

    try:
        with st.spinner("Agent is working — this takes 30–60 seconds..."):
            result = agent_executor.invoke({"input": task})

        steps = result.get("intermediate_steps", [])

        if steps:
            with st.expander(f"🔍 {len(steps)} searches performed", expanded=False):
                for i, (action, observation) in enumerate(steps):
                    query = action.tool_input.strip().replace('"', '').replace("'", "")
                    found = "❌ No results" if "No results" in str(observation) else "✅ Found"
                    st.markdown(f"**Search {i+1}:** {query} — {found}")

        st.divider()
        final_answer = result.get("output", "")

        if not final_answer or "Agent stopped" in final_answer:
            st.warning("Agent hit iteration limit. Increase Max Iterations in the sidebar and try again.")
        else:
            st.markdown(final_answer)

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.info("If you see a rate limit error, wait a few minutes and try again.")

elif run_button and not company:
    st.warning("Please enter a company name first.")
