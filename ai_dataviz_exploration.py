import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import math

# Helper: convert fig -> PNG bytes
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ==============
# 0) Streamlit setup
# ==============
st.set_page_config(page_title="AI Data Viz Assistant", layout="wide")
st.title("📊 AI-Powered Visual Exploration of Corporate Culture Data")
st.markdown(
    """
    <p style="font-size:1.1rem; margin-top:-0.25rem;">
      Using the dataset from
      <a href="https://www.culturex.com" target="_blank" rel="noopener noreferrer">CultureX</a>
      with corporate culture values across companies from different industries, as measured through
      anonymous Glassdoor reviews between Jan 1, 2023 and Apr 4, 2025.
    </p>
    """,
    unsafe_allow_html=True,
)


# --- Keep API key as is (note: hard-coding keys is risky in production) ---
# for local dev
# from dotenv import load_dotenv
# import os
# load_dotenv()  
# api_key = os.getenv("OPENAI_API_KEY")
# for deployment (e.g., Streamlit Cloud)
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.warning("Please provide a valid OpenAI API key in st.secrets as OPENAI_API_KEY to continue.")
    st.stop()

# ==============
# 1) Models
# ==============
llm_plan = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1,
    openai_api_key=api_key
)

llm_exec = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1,
    openai_api_key=api_key
)

llm_narrative = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1,
    openai_api_key=api_key
)

llm_explainer = ChatOpenAI(
    model="gpt-5-mini",
    temperature=1,
    openai_api_key=api_key
)

# ==============
# 2) Data source (hard-coded single CSV, no upload)
# ==============
CSV_PATH = "culturex_corporate_culture_data.csv"   

try:
    df = pd.read_csv(CSV_PATH)
    st.subheader("📄 Data Overview")
    page_size = 5
    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / page_size))

    # simple pagination control (no slider)
    page = st.number_input("Page number", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size

    st.caption(f"Showing rows {start+1}–{min(end, total_rows)} of {total_rows} (page {page} of {total_pages})")
    st.dataframe(df.iloc[start:end], use_container_width=True)
except FileNotFoundError:
    st.error(f"CSV not found at: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# ==============
# 2a) Hard-coded dataset context (hidden from UI)
# ==============
DATA_CONTEXT = """
The data show companies' scores on corporate culture values as measured through anonymous Glassdoor reviews. 
The scores are on z-score scale. 
The toxic culture scores were already reverse-coded.
The data include columns with company name, company description, and industry, and the following culture dimensions: work-life balance, transparency, toxic culture, supportive culture, strategy, leadership, innovation, open to feedback, and agility.
"""
data_context = DATA_CONTEXT  # <-- used internally; no UI element

instructions = st.text_area(
    "Enter your question or instructions for data visualization:",
    "Example: Show me relationship between innovation and agility."
)

# ==============
# 3) Shared state for LangGraph
# ==============
class VizState(TypedDict):
    schema: str
    instructions: str
    data_context: str
    plan: Optional[str]
    code: Optional[str]
    explanation: Optional[str]
    df: Optional[object]
    fig: Optional[object]
    error: Optional[str]
    narrative_code: Optional[str]
    narrative_text: Optional[str]
    retry_count_exec: int
    retry_count_narrative: int

# ==============
# 4) Helper to run code safely
# ==============
def run_exec(code: str, df: pd.DataFrame) -> plt.Figure:
    safe_code = (
        code.replace("display(fig)", "")
            .replace("plt.show()", "")
            .replace("display(", "# display(")
    )
    exec_env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(safe_code, exec_env)
    return exec_env["fig"]

def run_narrative(code: str, df: pd.DataFrame) -> str:
    exec_env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(code, exec_env)
    return exec_env["narrative"]

# ==============
# 5) Nodes with spinners
# ==============
def planner_node(state: VizState) -> VizState:
    with st.spinner("📝 Generating plan..."):
        plan_msg = llm_plan.invoke(f"""
            Dataset schema: {state['schema']}
            Dataset context: {state['data_context']}
            User request: {state['instructions']}

            Task: Break down the steps for both data wrangling (pandas) and visualization (seaborn + matplotlib) that would fulfill the user’s request - for example, creating a data visualization, answering a question, providing insights, or helping test a hypothesis.
            Requirements:
            - Wrangling must operate on the input DataFrame df
            - There can be multiple plots or subplots, but they must all be contained in a single matplotlib Figure
            - Visualization must end with a matplotlib Figure object called fig
            - Provide a single, concise step plan that best achieves the user’s request or answers their question.
            - Make the analysis as simple as possible while still being effective.
            - Do more complex analysis only if it clearly adds value or if your explictly asked to do so by a user.
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
            - When the user's request contains specific company names or industries, correct potential typos and then use key-word search to find relevant companies or industries in the dataset as people may misspell them.
            """)
        state["plan"] = plan_msg.content
    return state

class ExecCode(BaseModel):
    code: str

exec_llm = llm_exec.with_structured_output(ExecCode)

def executor_node(state: VizState) -> VizState:
    with st.spinner("⚙️ Generating and executing code..."):
        exec_plan = exec_llm.invoke(f"""
            Context plan:
            {state['plan']}

            The input DataFrame is named df and is already loaded.
            Write Python code that:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
            - Prefer simplicity and clarity over complexity
            - You may create multiple plots or subplots if it enhances the analysis and user's understanding, but they must all be contained in a single matplotlib Figure
            - Keep in mind that the generated plot should be well readable on the common computer screen (not too small, not too big)
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Always end with `fig` defined as the final Figure object
        """)
        code = exec_plan.code
        state["code"] = code

        try:
            state["fig"] = run_exec(code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

def repair_exec_node(state: VizState) -> VizState:
    if not state["error"]:
        return state

    state["retry_count_exec"] += 1
    if state["retry_count_exec"] > 3:
        return state

    with st.spinner(f"🔧 Repairing failed visualization code (attempt {state['retry_count_exec']}/3)..."):
        repair_msg = exec_llm.invoke(f"""
            The following visualization code failed with an error:
            ```
            {state['code']}
            ```
            Error message:
            {state['error']}

            Please suggest corrected Python code that fixes this issue.
            Constraints:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - You may create multiple plots or subplots if it enhances the analysis and user's understanding, but they must all be contained in a single matplotlib Figure
            - Prefer simplicity and clarity over complexity
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Keep in mind that the generated plot should be well readadble (not too small, not too crowded)
            - Always end with `fig` defined as the final Figure object
        """)
        repaired_code = repair_msg.code
        state["code"] = repaired_code

        try:
            state["fig"] = run_exec(repaired_code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

class NarrativeCode(BaseModel):
    code: str

narrative_llm = llm_narrative.with_structured_output(NarrativeCode)

def narrative_node(state: VizState) -> VizState:
    with st.spinner("📜 Generating narrative code..."):
        narrative_plan = narrative_llm.invoke(f"""
            User request: {state['instructions']}
            Dataset schema: {state['schema']}
            Dataset context: {state['data_context']}
            Context plan: {state['plan']}
            Visualization code:
            ```
            {state['code']}
            ```
            
            Task: Write Python code that generates a string variable named `narrative`.
            Requirements:
            - Only use column names from the dataset schema.
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - Only return runnable Python code.
        """)

        state["narrative_code"] = narrative_plan.code

        try:
            state["narrative_text"] = run_narrative(state["narrative_code"], state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

def repair_narrative_node(state: VizState) -> VizState:
    if not state["error"]:
        return state

    state["retry_count_narrative"] += 1
    if state["retry_count_narrative"] > 3:
        return state

    with st.spinner(f"🔧 Repairing failed narrative code (attempt {state['retry_count_narrative']}/3)..."):
        repair_msg = narrative_llm.invoke(f"""
            The following narrative code failed with an error:
            ```
            {state['narrative_code']}
            ```
            Error message:
            {state['error']}

            Please suggest corrected Python code that fixes this issue.
            Constraints:
            - Only use column names from the dataset schema.
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - Only return runnable Python code.
        """)
        repaired_code = repair_msg.code
        state["narrative_code"] = repaired_code

        try:
            state["narrative_text"] = run_narrative(repaired_code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

def explainer_node(state: VizState) -> VizState:
    with st.spinner("💬 Generating explanation..."):
        explain_msg = llm_explainer.invoke(f"""
            User request: {state['instructions']}
            Dataset context: {state['data_context']}
            Context plan:
            {state['plan']}

            Narrative string:
            {state['narrative_text']}

            Task: Create a narrative explanation of what the generated chart(s) show and how to interpret them,
            and provide specific insights revealed by the analysis for a non-technical audience.
            Constraints:
            - Make it concise and clear.
            - Do not output code. Write only text.
        """)
        state["explanation"] = explain_msg.content.strip()
    return state

# ==============
# 6) Build the workflow graph
# ==============
workflow = StateGraph(VizState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("repair_exec", repair_exec_node)
workflow.add_node("narrative", narrative_node)
workflow.add_node("repair_narrative", repair_narrative_node)
workflow.add_node("explainer", explainer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")

workflow.add_conditional_edges(
    "executor",
    lambda state: "repair_exec" if state.get("error") else "narrative"
)

workflow.add_conditional_edges(
    "repair_exec",
    lambda state: "executor" if state.get("error") and state["retry_count_exec"] <= 3 else "narrative"
)

workflow.add_conditional_edges(
    "narrative",
    lambda state: "repair_narrative" if state.get("error") else "explainer"
)

workflow.add_conditional_edges(
    "repair_narrative",
    lambda state: "narrative" if state.get("error") and state["retry_count_narrative"] <= 3 else "explainer"
)

workflow.add_edge("explainer", END)

app = workflow.compile()

# ==============
# 7) Streamlit integration
# ==============
if "viz_result" not in st.session_state:
    st.session_state.viz_result = None

if df is not None and instructions and st.button("Generate Visualization"):
    schema_str = ", ".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items())

    state: VizState = {
        "schema": schema_str,
        "instructions": instructions,
        "data_context": data_context,
        "df": df,
        "plan": None,
        "code": None,
        "fig": None,
        "explanation": None,
        "error": None,
        "narrative_code": None,
        "narrative_text": None,
        "retry_count_exec": 0,
        "retry_count_narrative": 0,
    }

    result = app.invoke(state)

    # 🔑 Convert fig to PNG bytes if it exists
    if result["fig"]:
        result["fig_png"] = fig_to_png_bytes(result["fig"])
    else:
        result["fig_png"] = None

    st.session_state.viz_result = result

# ---- Show persisted results (if any) ----
if st.session_state.viz_result:
    result = st.session_state.viz_result

    if result.get("fig_png"):
        st.image(result["fig_png"], use_container_width=True)

    if result.get("explanation"):
        st.subheader("💡 Explanation")
        st.write(result["explanation"])

    if result.get("error"):
        st.error(f"Execution still failing: {result['error']}")