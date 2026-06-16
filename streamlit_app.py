import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

# load_dotenv()  # only needed locally — Hugging Face injects secrets automatically

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Company Research Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Company Research Agent")
st.write("Powered by LangChain + Groq (Llama 3.3) · Enter a company name and the agent researches it autonomously.")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_iter = st.slider(
        "Max agent iterations",
        min_value=3,
        max_value=15,
        value=8,
        help="How many steps the agent can take. More = thorough but slower."
    )
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. You enter a company name")
    st.markdown("2. Agent decides what to search")
    st.markdown("3. Searches web multiple times")
    st.markdown("4. Writes a professional briefing")
    st.divider()
    st.markdown("Built with LangChain · Groq · Streamlit")

# ── Main input ────────────────────────────────────────────────────────
company = st.text_input(
    "Enter a company name:",
    placeholder="e.g. Apple, Stripe, Ryanair, Revolut..."
)

run_button = st.button("🔍 Research this company", type="primary")

# ── Only run when button is clicked and company is entered ────────────
if run_button and company:

    # ── Set up the LLM ────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # ── Set up the search tool ────────────────────────────────────────
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", time="m", max_results=5)
    search = DuckDuckGoSearchRun(api_wrapper=wrapper)

    tools = [
        Tool(
            name="web_search",
            func=lambda q: search.run(q.strip().strip('"').strip("'")),
            description="""Use this to search the web for current information.
            Input should be a plain search query with NO quotes around it.
            Example: Bata company overview
            NOT: "Bata company overview"
            Use this when you need recent news, financial data,
            or any facts about a company or topic."""
        )
    ]

    # ── Load the ReAct prompt ─────────────────────────────────────────
    prompt = hub.pull("hwchase17/react")

    # ── Build the agent ───────────────────────────────────────────────
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,        # we handle display ourselves below
        max_iterations=max_iter,
        handle_parsing_errors=True,
        return_intermediate_steps=True   # lets us show the thinking steps
    )

    # ── Task definition ───────────────────────────────────────────────
    task = f"""
    Research '{company}' and write a professional analyst briefing.
    
    Use web_search to find information. Search without quotes.
    Good search example: {company} company overview
    Bad search example: "{company} company overview"
    
    Write the final briefing in this exact format:

    ## {company} — Analyst Briefing

    **Overview**
    [2 sentences on what the company does]

    **Recent Developments**
    [2-3 bullet points of recent news]

    **Financial Highlights**
    [Key revenue, growth, or funding figures]

    **Key Risk**
    [One main challenge or risk]

    **Outlook**
    [One sentence — growing, stable, or under pressure]

    Keep it concise, factual, and professional.
    Only include information you actually found — do not make up figures.
    """

    # ── Run with live status updates ──────────────────────────────────
    with st.status(f"🔍 Researching {company}...", expanded=True) as status:
        st.write("Agent is starting up...")

        try:
            result = agent_executor.invoke({"input": task})

            # Show the thinking steps inside the status box
            steps = result.get("intermediate_steps", [])
            for i, (action, observation) in enumerate(steps):
                st.write(f"**Step {i+1}:** Searched for `{action.tool_input}`")
                st.caption(f"Found: {str(observation)[:150]}...")

            status.update(
                label=f"✅ Research complete — {len(steps)} searches performed",
                state="complete"
            )

        except Exception as e:
            status.update(label="❌ Something went wrong", state="error")
            st.error(f"Error: {str(e)}")
            st.stop()

    # ── Display the final briefing ────────────────────────────────────
    st.divider()
    st.subheader(f"📋 Analyst Briefing: {company}")

    final_answer = result.get("output", "No output generated.")

    if "Agent stopped" in final_answer:
        st.warning("The agent hit its iteration limit before finishing. Try increasing Max Iterations in the sidebar.")
    else:
        # Clean up the output — remove any leftover quotes or formatting
        cleaned = final_answer.strip().strip('"')
        st.markdown(cleaned)

    # ── Show full thinking log in expander ────────────────────────────
    with st.expander("🧠 See agent thinking log (optional)", expanded=False):
        steps = result.get("intermediate_steps", [])
        for i, (action, observation) in enumerate(steps):
            st.markdown(f"**🔍 Search {i+1}:** `{action.tool_input}`")
            st.markdown(f"**📄 Result:** {str(observation)[:500]}...")
            st.divider()

elif run_button and not company:
    st.warning("Please enter a company name first.")