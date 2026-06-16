import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(
    page_title="Company Research Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Company Research Agent")
st.write("Enter a company name — the agent searches the web autonomously and writes a professional briefing.")

# ── Sidebar ──────────────────────────────────────────────────────────
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
    st.markdown("Built with LangChain · Groq · Streamlit")

# ── Input ────────────────────────────────────────────────────────────
company = st.text_input(
    "Enter a company name:",
    placeholder="e.g. Apple, Ryanair, Revolut, Bata..."
)
run_button = st.button("🔍 Research this company", type="primary")

if run_button and company:

    # ── LLM setup ────────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # ── Tool setup ───────────────────────────────────────────────────
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="web_search",
            func=lambda q: search.run(q.replace('"', '').replace("'", "")),
            description="""Search the web for information about a company.
            Always use plain text queries with no quotes.
            Good: Bata company overview
            Bad: "Bata company overview"
            Search multiple times for different aspects."""
        )
    ]

    # ── Agent setup ──────────────────────────────────────────────────
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

    # ── Task ─────────────────────────────────────────────────────────
    task = f"""Research the company {company} and write a professional analyst briefing.

Search for: overview, recent news, financial highlights, risks.
Use plain text searches without any quote marks.

Write your final answer in exactly this format:

**{company} — Analyst Briefing**

**What they do**
Write 2 sentences describing the company.

**Recent developments**
- Point 1
- Point 2
- Point 3

**Financial highlights**
Write 2 sentences with key figures you found.

**Key risk**
Write 1 sentence on their biggest challenge.

**Overall outlook**
Write 1 sentence — growing, stable, or under pressure.

Use only facts you found. Do not invent figures."""

    # ── Live search progress ──────────────────────────────────────────
    st.divider()
    st.subheader(f"🔍 Researching {company}...")
    progress_container = st.container()

    try:
        with st.spinner("Agent is working..."):
            result = agent_executor.invoke({"input": task})

        steps = result.get("intermediate_steps", [])

        # Show what was searched
        with progress_container:
            st.success(f"✅ Completed — {len(steps)} searches performed")
            for i, (action, observation) in enumerate(steps):
                query = action.tool_input.replace('"', '').replace("'", "")
                found = "No results" if "No good" in str(observation) else "✓ Found results"
                st.caption(f"Search {i+1}: {query} — {found}")

        # ── Final briefing ────────────────────────────────────────────
        st.divider()
        final_answer = result.get("output", "No output generated.")

        if "Agent stopped" in final_answer:
            st.warning("Agent hit iteration limit. Increase Max Iterations in the sidebar and try again.")
        else:
            st.markdown(final_answer)

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

elif run_button and not company:
    st.warning("Please enter a company name first.")