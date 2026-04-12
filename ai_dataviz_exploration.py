"""
AI-Powered Visual Exploration of Corporate Culture Data
Multi-agent orchestration via LiteLLM — model-agnostic, no LangChain/LangGraph.

Agent pipeline:
  1. Clarifier        — asks user 1-2 targeted follow-up questions
  2. Planner          — builds analysis + viz plan
  3. Critic (loop)    — critiques plan, up to 3 rounds of plan↔critic refinement
  4. Executor         — generates + runs matplotlib/seaborn code (auto-repair up to 3x)
  5. Narrative        — computes statistics and writes data-driven narrative (auto-repair up to 3x)
  6. Explainer        — synthesizes final insight for non-technical audience
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, json, math, os, re, textwrap
from dataclasses import dataclass, field
from typing import Optional

from litellm import completion
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 0.  Page config & styling
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Data Viz Assistant", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.02em;
}

.stApp {
    background: #f7f5f2;
    color: #1a1a1a;
}

section[data-testid="stSidebar"] {
    background: #eeecea;
    border-right: 1px solid #d8d4cf;
}

.agent-card {
    background: #ffffff;
    border: 1px solid #d8d4cf;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.3s;
}

.agent-card.active {
    border-color: #9b6f2f;
    box-shadow: 0 0 20px rgba(155,111,47,0.08);
}

.agent-card.done {
    border-color: #4a7c4a;
}

.agent-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.2rem;
}

.agent-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #1a1a1a;
}

.critic-badge {
    display: inline-block;
    background: #fdf6ec;
    border: 1px solid #9b6f2f44;
    color: #9b6f2f;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    margin-left: 0.5rem;
    vertical-align: middle;
}

.insight-box {
    background: linear-gradient(135deg, #fffdf9 0%, #fdf8f2 100%);
    border: 1px solid #d8d4cf;
    border-left: 3px solid #9b6f2f;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}

.stButton > button {
    background: #9b6f2f;
    color: #ffffff;
    font-weight: 500;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #b8873a;
    transform: translateY(-1px);
}

.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    border: 1px solid #d8d4cf !important;
    color: #1a1a1a !important;
    border-radius: 8px !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #9b6f2f !important;
    box-shadow: 0 0 0 1px #9b6f2f44 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #d8d4cf;
    border-radius: 8px;
    overflow: hidden;
}

.pipeline-header {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #d8d4cf;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="margin-bottom:0.1rem;">📊 AI Data Viz Assistant</h1>
<p style="color:#666; font-size:1rem; margin-top:0.25rem;">
  Corporate culture data from
  <a href="https://www.culturex.com" target="_blank" style="color:#c9a84c; text-decoration:none;">CultureX</a>
  · Glassdoor reviews Jan 2023 – Apr 2025
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 1.  API key
# ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

if not api_key or not api_key.startswith("AIza"):
    st.warning("Set GEMINI_API_KEY in .env or Streamlit secrets.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key
MODEL = "gemini/gemini-3-flash-preview"   # LiteLLM prefix/model name

# ─────────────────────────────────────────────
# 2.  Load data
# ─────────────────────────────────────────────
CSV_PATH = "culturex_corporate_culture_data.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV not found at: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

DATA_CONTEXT = """
The dataset contains companies' scores on corporate culture dimensions as measured through 
anonymous Glassdoor reviews (z-score scale; toxic culture is reverse-coded so higher = less toxic).
Columns: company name, company description, industry, and culture dimensions:
work_life_balance, transparency, toxic_culture, supportive_culture, strategy, 
leadership, innovation, open_to_feedback, agility.
"""

# Normalise column names to snake_case at load time so generated code never
# contains spaces, hyphens, or other characters that cause string-wrapping issues.
def _to_snake(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)   # spaces, hyphens, slashes → underscore
    s = re.sub(r"[^\w]", "", s)         # drop anything else non-word
    s = re.sub(r"_+", "_", s).strip("_")
    return s

original_columns = df.columns.tolist()
df.columns = [_to_snake(c) for c in original_columns]

# Rich schema: exact column names + dtypes + sample values.
# This is injected into every agent prompt to prevent column name hallucination.
def _build_schema_str(dataframe):
    lines = ["EXACT COLUMN NAMES (use these verbatim, no substitutions):"]
    for col, dtype in dataframe.dtypes.items():
        sample_vals = dataframe[col].dropna().head(3).tolist()
        sample_str = ", ".join(repr(v) for v in sample_vals)
        lines.append(f"  - {col!r}  ({dtype})  e.g. {sample_str}")
    return "\n".join(lines)

schema_str = _build_schema_str(df)

# ─────────────────────────────────────────────
# 3.  Helpers: LiteLLM wrappers
# ─────────────────────────────────────────────
def call_llm(system: str, messages: list, max_tokens: int = 1200) -> str:
    """Call any LiteLLM-supported model. Prepends system prompt; returns text content."""
    response = completion(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return response.choices[0].message.content.strip()


def call_llm_raw(system: str, messages: list, max_tokens: int = 1200):
    """Like call_llm but returns the full response so callers can inspect finish_reason."""
    response = completion(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return response


def _extract_code_block(text: str, was_truncated: bool = False) -> str:
    """Extract Python code from a model response robustly.

    Tries, in order:
    1. ```python ... ``` fenced block
    2. ``` ... ``` fenced block (any language tag)
    3. Heuristic: strip any non-code preamble lines before the first import/def/assignment
    4. Return text as-is (last resort)

    If was_truncated is True and no closing ``` is found, raises ValueError
    instead of returning broken code.
    """
    # Strategy 1 & 2: fenced block (complete — has closing ```)
    m = re.search(r"```(?:[a-zA-Z]*)\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Check for an OPEN fenced block with no closing — strong truncation signal
    open_fence = re.search(r"```(?:[a-zA-Z]*)\n?(.*)", text, re.DOTALL)
    if open_fence:
        if was_truncated:
            raise ValueError(
                "LLM response was truncated (hit max_tokens) mid-code-block. "
                "Retry with a higher token budget."
            )
        # Even if finish_reason was 'stop', an unclosed fence is suspicious.
        # Still try to use it but log a warning.
        return open_fence.group(1).strip()

    # Strategy 3: find first line that looks like Python and take everything from there
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("def ")
                or stripped.startswith("fig ")
                or stripped.startswith("df ")
                or re.match(r"^[a-zA-Z_][\w]* *=", stripped)):
            return "\n".join(lines[i:]).strip()

    return text.strip()


def call_llm_json(system: str, messages: list, max_tokens: int = 1200) -> dict:
    """Like call_llm but instructs the model to return JSON and parses the result.

    Retries up to 3 times. On each failure strips control chars and widens
    the extraction strategy before retrying.
    """
    system_json = system + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown fences, no preamble."
    last_err = None
    for attempt in range(3):
        try:
            raw = call_llm(system_json, messages, max_tokens)
            clean = raw.strip()
            # Strip markdown fences
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()
            # Strip control characters (common Gemini quirk)
            clean = re.sub(r"[\x00-\x1f\x7f](?![\n\r\t])", "", clean)
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            continue
    raise ValueError(f"call_llm_json failed after 3 attempts: {last_err}")


def call_llm_code(system: str, messages: list, max_tokens: int = 2500) -> str:
    """Ask the model to return a Python code block; extract robustly without JSON.

    This avoids JSON parse failures caused by quotes and special characters inside
    generated code. The model is asked to wrap code in triple backticks; we extract
    it with a regex and fall back to returning the full response if no block is found.

    Key reliability feature: if the response is truncated (finish_reason == 'length'),
    automatically retries with a larger token budget (up to 3 attempts, escalating
    from max_tokens → 1.5x → 2x).
    """
    system_code = (
        system
        + "\n\nReturn ONLY a Python code block wrapped in triple backticks like this:\n"
        + "```python\n# your code here\n```\n"
        + "No JSON, no explanation outside the code block.\n"
        + "IMPORTANT: Keep code concise. Use list comprehensions and compact variable names.\n"
        + "Avoid unnecessary comments. This helps ensure the full code fits in the response."
    )

    token_budgets = [max_tokens, int(max_tokens * 1.5), max_tokens * 2]

    for attempt, budget in enumerate(token_budgets):
        response = call_llm_raw(system_code, messages, budget)
        raw = response.choices[0].message.content.strip()
        finish_reason = getattr(response.choices[0], "finish_reason", "stop") or "stop"
        was_truncated = finish_reason == "length"

        try:
            code = _extract_code_block(raw, was_truncated=was_truncated)
            # Extra safety: try to compile before returning
            compile(code, "<llm_output>", "exec")
            return code
        except (ValueError, SyntaxError) as e:
            if attempt < len(token_budgets) - 1 and was_truncated:
                # Truncation caused the error — retry with more tokens
                continue
            # Last attempt or not a truncation issue — return what we have
            # and let the caller's run_exec / _check_syntax handle the error
            return _extract_code_block(raw, was_truncated=False)


# ─────────────────────────────────────────────
# 4.  Helper: exec code safely
# ─────────────────────────────────────────────
def _auto_fix_truncation(code: str) -> str:
    """Attempt to fix code that was truncated mid-output.

    Handles common cases:
    - Unclosed triple-quoted strings → close them
    - Unclosed string literals on the last line → remove the broken last line
    - Unclosed brackets/parens → close them
    - Missing `fig` assignment after subplots → append `fig = plt.gcf()`

    Returns the (possibly repaired) code. Does NOT raise — caller should still
    compile-check the result.
    """
    lines = code.rstrip().splitlines()
    if not lines:
        return code

    # 0. Handle unclosed triple-quoted strings FIRST (multi-line problem).
    #    Count occurrences of ''' and """; if odd, the string was truncated.
    for triple in ['"""', "'''"]:
        count = code.count(triple)
        if count % 2 != 0:
            # Find the line where the opening triple-quote starts (unmatched)
            # and try to close it, or strip everything from the opening onwards.
            # Strategy: try closing with the triple-quote first.
            test_code = code + "\n" + triple
            try:
                compile(test_code, "<auto_fix_triple>", "exec")
                code = test_code
                lines = code.rstrip().splitlines()
                break  # fixed
            except SyntaxError:
                # Closing didn't work (e.g. broken expression around it).
                # Fall back: remove everything from the last unmatched opening.
                last_open = code.rfind(triple)
                if last_open > 0:
                    code = code[:last_open].rstrip()
                    # Also remove the incomplete assignment line (e.g. `narrative = `)
                    code_lines = code.rstrip().splitlines()
                    while code_lines and code_lines[-1].rstrip().endswith("="):
                        code_lines.pop()
                    code = "\n".join(code_lines)
                    lines = code.rstrip().splitlines()
                break

    # 1. If code doesn't compile, pop lines from the end until it does (or we run out).
    #    This handles truncated strings, unclosed brackets, etc. more robustly than
    #    counting quotes per line (which fails when many quoted items are on one line).
    #    We pop at most 10 lines to avoid stripping valid code.
    pops = 0
    while lines and pops < 10:
        test = "\n".join(lines)
        try:
            compile(test, "<auto_fix_pop>", "exec")
            break  # compiles — stop popping
        except SyntaxError:
            lines.pop()
            pops += 1

    code = "\n".join(lines)

    # 2. Try to compile; if still broken with unclosed brackets at EOF, close them
    for _ in range(5):
        try:
            compile(code, "<auto_fix>", "exec")
            break
        except SyntaxError as e:
            msg = str(e.msg).lower() if e.msg else ""
            if "eof" in msg or "unterminated" in msg or "unexpected" in msg:
                # Try closing the most likely bracket
                if code.count("(") > code.count(")"):
                    code += "\n)"
                elif code.count("[") > code.count("]"):
                    code += "\n]"
                elif code.count("{") > code.count("}"):
                    code += "\n}"
                else:
                    break
            else:
                break

    # 3. Ensure `fig` is defined (common miss when code gets truncated after plt calls)
    if "fig" not in code.split("=")[0] if "=" in code else True:
        try:
            compile(code, "<auto_fix>", "exec")
        except SyntaxError:
            pass
        else:
            # Code compiles but may lack `fig = ...`
            if not re.search(r"\bfig\s*[,=]", code):
                code += "\nfig = plt.gcf()"

    return code


def _check_syntax(code: str) -> None:
    """Raise SyntaxError with the first 300 chars of code shown, for easier debugging."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        preview = code[:300].replace("\n", "\\n")
        raise SyntaxError(
            f"{e.msg} (line {e.lineno})\n"
            f"Code preview: {preview!r}"
        ) from None


def run_exec(code: str, df: pd.DataFrame) -> plt.Figure:
    safe = (code
            .replace("display(fig)", "")
            .replace("plt.show()", "")
            .replace("display(", "# display("))
    # Try auto-fixing truncation artifacts before raising
    safe = _auto_fix_truncation(safe)
    _check_syntax(safe)
    env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(safe, env)  # noqa: S102
    return env["fig"]


def run_narrative(code: str, df: pd.DataFrame) -> str:
    code = _auto_fix_truncation(code)
    _check_syntax(code)
    env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(code, env)  # noqa: S102
    return env["narrative"]


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# 5.  State
# ─────────────────────────────────────────────
@dataclass
class VizState:
    instructions: str
    clarifying_questions: list = field(default_factory=list)
    clarifying_answers: str = ""
    plan: str = ""
    critic_feedback: list = field(default_factory=list)   # list of strings
    code: str = ""
    fig: Optional[object] = None
    fig_png: Optional[bytes] = None
    narrative_code: str = ""
    narrative_text: str = ""
    explanation: str = ""
    error: Optional[str] = None
    retry_exec: int = 0
    retry_narrative: int = 0
    prior_context: str = ""   # summary of previous turns for follow-up questions


def _build_prior_context(history: list) -> str:
    """Build a concise prior-context string from conversation history.

    Each entry in history is a dict with keys: instructions, clarifying_answers,
    plan, code, narrative_text, explanation, fig_png (ignored here).

    Strategy for long conversations:
    - Last 3 turns: full detail (plan, code, narrative, explanation)
    - Older turns: compact summary (just instructions + clarifying answers + key insight)
    This keeps context useful without bloating the prompt on turn 6+.
    """
    if not history:
        return ""

    parts = ["CONVERSATION HISTORY (previous analysis turns):"]
    n = len(history)

    for i, turn in enumerate(history, 1):
        parts.append(f"\n--- Turn {i} ---")
        parts.append(f"User request: {turn['instructions']}")

        # Always include clarifying answers — these contain critical user preferences
        if turn.get("clarifying_answers"):
            parts.append(f"User's clarifications:\n{turn['clarifying_answers']}")

        if i > n - 3:
            # Recent turns: full detail
            if turn.get("plan"):
                plan_preview = turn["plan"][:600]
                parts.append(f"Plan summary: {plan_preview}")
            if turn.get("code"):
                code_preview = turn["code"][:800]
                parts.append(f"Visualization code:\n```python\n{code_preview}\n```")
            if turn.get("narrative_text"):
                parts.append(f"Statistical narrative: {turn['narrative_text'][:400]}")
            if turn.get("explanation"):
                parts.append(f"Insight: {turn['explanation'][:400]}")
        else:
            # Older turns: compact — just the key insight
            if turn.get("explanation"):
                parts.append(f"Insight: {turn['explanation'][:200]}")

    parts.append("\n--- END OF HISTORY ---")
    return "\n".join(parts)


# ─────────────────────────────────────────────
# 6.  Agent nodes
# ─────────────────────────────────────────────


def clarifier_agent(instructions: str, history: list = None) -> list[str]:
    """Returns 2-3 targeted clarifying questions (or empty list if none needed).

    Uses plain-text output instead of JSON to avoid parse failures on simple responses.
    The model outputs one question per line, or the literal word NONE if no questions needed.
    Retries with a higher token budget if the response is truncated.

    If history is provided (follow-up questions), the clarifier is aware of what
    the user already specified and will NOT re-ask those questions.
    """
    history_context = ""
    if history:
        parts = []
        for i, turn in enumerate(history, 1):
            turn_parts = [f"Turn {i} — Request: {turn.get('instructions', '')}"]
            if turn.get("clarifying_answers"):
                turn_parts.append(f"User's clarifications: {turn['clarifying_answers']}")
            if turn.get("plan"):
                turn_parts.append(f"Analysis plan: {turn['plan'][:400]}")
            if turn.get("explanation"):
                turn_parts.append(f"Insight: {turn['explanation'][:200]}")
            parts.append("\n".join(turn_parts))
        if parts:
            history_context = (
                "\n\nCONVERSATION HISTORY — the user has already established these details "
                "in previous turns. Do NOT ask about anything already covered here "
                "(companies, metrics, chart types, comparisons, etc.):\n\n"
                + "\n---\n".join(parts)
                + "\n\nOnly ask about NEW ambiguities introduced by the latest request. "
                "If the latest request is clear given the history above, reply NONE."
            )

    system_prompt = textwrap.dedent(f"""
        You are a clarifier for a data analysis assistant.
        Dataset context: {DATA_CONTEXT}
        Schema: {schema_str}
        {history_context}

        Your job: decide if the user's request is ambiguous or under-specified in ways
        that would meaningfully change the analysis. If so, write exactly 2 or 3 SHORT
        clarifying questions, one per line. Always write at least 2 questions.
        Keep each question under 20 words.
        Think about what specifics would sharpen the analysis: which companies,
        which metrics, what comparison, what time scope, what chart type, etc.

        If the request is completely unambiguous and no clarification would help,
        reply with just: NONE

        Format rules:
        - One question per line
        - No numbering, no bullet points, no prefixes
        - No other text — only the questions or the word NONE
    """)

    for budget in [400, 600]:
        response = call_llm_raw(
            system_prompt,
            [{"role": "user", "content": instructions}],
            max_tokens=budget,
        )
        raw = response.choices[0].message.content.strip()
        finish_reason = getattr(response.choices[0], "finish_reason", "stop") or "stop"

        if finish_reason != "length":
            break
        # Truncated — retry with more tokens

    stripped = raw.strip()
    if not stripped or stripped.upper() == "NONE":
        return []
    # Split on newlines, drop blanks, strip any numbering/bullets the LLM may add
    questions = []
    for line in stripped.splitlines():
        q = line.strip()
        if not q:
            continue
        # Strip leading numbering like "1.", "1)", "- ", "* "
        q = re.sub(r"^[\d]+[.)]\s*", "", q)
        q = re.sub(r"^[-*•]\s*", "", q)
        q = q.strip()
        if q and q.upper() != "NONE":
            questions.append(q)

    # If the last question looks cut off (no punctuation at end), drop it
    if questions and not questions[-1].rstrip().endswith(("?", ".", "!", ")")):
        questions.pop()

    return questions


def planner_agent(state: VizState) -> str:
    context = f"User clarifications: {state.clarifying_answers}" if state.clarifying_answers else ""
    prior = f"\n{state.prior_context}\n" if state.prior_context else ""
    return call_llm(
        system=textwrap.dedent(f"""
            You are a senior data analyst and visualization expert.
            Dataset schema: {schema_str}
            Dataset context: {DATA_CONTEXT}
            {context}
            {prior}

            Task: Break down the steps for both data wrangling (pandas) and visualization
            (seaborn + matplotlib) that would fulfill the user's request — for example,
            creating a data visualization, answering a question, providing insights, or
            helping test a hypothesis.
            If there is conversation history above, the user may be asking a follow-up
            question. In that case, build on the previous analysis where it makes sense —
            reuse relevant data wrangling, refine the visualization, or take a new angle
            as the user's question requires.
            Requirements:
            - Wrangling must operate on the input DataFrame df
            - There can be multiple plots or subplots, but they must all be contained in a single matplotlib Figure
            - Visualization must end with a matplotlib Figure object called fig
            - Provide a single, concise step plan that best achieves the user's request or answers their question.
            - Make the analysis as simple as possible while still being effective.
            - Do more complex analysis only if it clearly adds value or if explicitly asked by the user.
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear
              and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid
              distortion or chartjunk.
            - When the user's request contains specific company names or industries, correct potential typos
              and then use keyword search to find relevant companies or industries in the dataset as people
              may misspell them.
            Output only the plan, no code.
        """),
        messages=[{"role": "user", "content": state.instructions}],
        max_tokens=2000,
    )


def critic_agent(state: VizState, round_num: int) -> tuple[bool, str]:
    """Returns (approved: bool, feedback: str).

    Uses plain-text output to avoid JSON parse failures on long feedback strings.
    Format: first line is APPROVED or REJECTED, remaining lines are the feedback.
    """
    history_str = ""
    if state.critic_feedback:
        history_str = "Previous critic rounds:\n" + "\n---\n".join(state.critic_feedback)

    raw = call_llm(
        system=textwrap.dedent(f"""
            You are a rigorous internal critic for a data analysis pipeline.
            Dataset schema: {schema_str}
            Dataset context: {DATA_CONTEXT}
            User request: {state.instructions}
            {history_str}

            Evaluate the proposed analysis plan:
            1. Does it answer the user's actual question?
            2. Are the visualization choices appropriate?
            3. Are column names valid per the schema?
            4. Is the logic coherent and sufficient?
            5. Is complexity justified?

            Reply in exactly this format — no extra text before or after:
            APPROVED
            <your feedback or 'Plan is solid.'>

            or:
            REJECTED
            <specific, actionable critique>

            Be strict for rounds 1-2, more lenient on round {round_num} (final round).
            Approve if the plan is genuinely good OR if issues are minor and it's round {round_num}.
        """),
        messages=[{"role": "user", "content": f"Round {round_num} plan:\n\n{state.plan}"}],
        max_tokens=1000,
    )
    lines = raw.strip().splitlines()
    verdict = lines[0].strip().upper() if lines else "APPROVED"
    feedback = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    approved = verdict == "APPROVED"
    return approved, feedback


def planner_revise_agent(state: VizState) -> str:
    """Planner incorporates latest critic feedback."""
    latest_feedback = state.critic_feedback[-1] if state.critic_feedback else ""
    return call_llm(
        system=textwrap.dedent(f"""
            You are a senior data analyst revising your analysis plan.
            Dataset schema: {schema_str}
            Dataset context: {DATA_CONTEXT}
            User request: {state.instructions}

            Your previous plan received this critic feedback:
            {latest_feedback}

            Revise the plan to address all feedback points.
            Output only the improved plan, no explanations.
        """),
        messages=[{"role": "user", "content": f"Previous plan:\n\n{state.plan}"}],
        max_tokens=2000,
    )


def executor_agent(state: VizState) -> tuple[Optional[plt.Figure], str, Optional[str]]:
    """Returns (fig, code, error)."""
    prior = f"\n{state.prior_context}\n" if state.prior_context else ""
    code = call_llm_code(
        system=textwrap.dedent(f"""
            You are a Python data visualization engineer.
            Generate executable Python code following the plan exactly.
            {prior}

            CRITICAL — COLUMN NAMES:
            {schema_str}
            You MUST use only the exact column names listed above. Do NOT invent, rename,
            or abbreviate any column name. If the plan mentions a column not in this list,
            use the closest exact match from the list above.

            The input DataFrame is named df and is already loaded.
            Write Python code that:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - Apply good data visualization principles: choose the right chart for the data,
              keep visuals clear and uncluttered, label everything, use accessible colors,
              highlight the key insight, and avoid distortion or chartjunk.
            - Prefer simplicity and clarity over complexity
            - You may create multiple plots or subplots if it enhances the analysis and
              user's understanding, but they must all be contained in a single matplotlib Figure
            - Keep in mind that the generated plot should be well readable on the common
              computer screen (not too small, not too crowded)
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Always end with `fig` defined as the final Figure object
        """),
        messages=[{"role": "user", "content": f"Plan:\n{state.plan}"}],
        max_tokens=2500,
    )
    try:
        fig = run_exec(code, df)
        return fig, code, None
    except Exception as e:
        return None, code, str(e)


def repair_exec_agent(state: VizState) -> tuple[Optional[plt.Figure], str, Optional[str]]:
    code = call_llm_code(
        system=textwrap.dedent(f"""
            You are a Python debugging expert. Fix the broken visualization code.

            CRITICAL — ONLY USE THESE EXACT COLUMN NAMES:
            {schema_str}
            If the error mentions a column not found, replace it with the correct name from
            the list above. Do NOT invent new column names.

            Please suggest corrected Python code that fixes this issue.
            Constraints:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - You may create multiple plots or subplots if it enhances the analysis and
              user's understanding, but they must all be contained in a single matplotlib Figure
            - Prefer simplicity and clarity over complexity
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Keep in mind that the generated plot should be well readable (not too small, not too crowded)
            - Always end with `fig` defined as the final Figure object
        """),
        messages=[{"role": "user", "content": f"Broken code:\n```python\n{state.code}\n```\nError: {state.error}"}],
        max_tokens=2500,
    )
    try:
        fig = run_exec(code, df)
        return fig, code, None
    except Exception as e:
        return None, code, str(e)


def narrative_agent(state: VizState) -> tuple[str, str, Optional[str]]:
    """Returns (narrative_text, narrative_code, error)."""
    code = call_llm_code(
        system=textwrap.dedent(f"""
            You are a statistical analyst. Write Python code that computes a data-driven
            narrative string from DataFrame `df`.
            Rules:
            - Only use column names from schema: {schema_str}
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - No return/display/print
            - Only return runnable Python code.
        """),
        messages=[{"role": "user", "content": (
            f"User request: {state.instructions}\n"
            f"Plan: {state.plan}\n"
            f"Viz code:\n```python\n{state.code}\n```"
        )}],
        max_tokens=2000,
    )
    try:
        narrative = run_narrative(code, df)
        return narrative, code, None
    except Exception as e:
        return "", code, str(e)


def repair_narrative_agent(state: VizState) -> tuple[str, str, Optional[str]]:
    code = call_llm_code(
        system=textwrap.dedent(f"""
            Fix the broken narrative code.
            Constraints:
            - Only use column names from schema: {schema_str}
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - Only return runnable Python code.
        """),
        messages=[{"role": "user", "content": f"Broken code:\n```python\n{state.narrative_code}\n```\nError: {state.error}"}],
        max_tokens=2000,
    )
    try:
        narrative = run_narrative(code, df)
        return narrative, code, None
    except Exception as e:
        return "", code, str(e)


def explainer_agent(state: VizState) -> str:
    """Generate a concise insight summary. Retries if the response is truncated."""
    prior = ""
    if state.prior_context:
        prior = (
            "\nThe user has been having an ongoing conversation about this data. "
            "Here is the prior context:\n" + state.prior_context + "\n"
            "Frame your explanation in the context of this ongoing exploration.\n"
        )
    system_prompt = textwrap.dedent(f"""
        You are a business insight communicator.
        {prior}
        Task: Create a narrative explanation of what the generated chart(s) show and
        how to interpret them, and provide specific insights revealed by the analysis
        for a non-technical audience.
        Constraints:
        - Keep it under 200 words. Be concise and clear.
        - Do not use bullet points or markdown formatting (no *, **, or - lists).
        - Write in flowing prose paragraphs only.
        - Do not output code. Write only text.
        - Make sure your response is complete — finish every sentence.
    """)
    user_content = (
        f"User request: {state.instructions}\n"
        f"Plan: {state.plan}\n"
        f"Statistical narrative: {state.narrative_text}"
    )

    # Try with escalating token budgets if truncated
    for budget in [1000, 1500]:
        response = call_llm_raw(
            system_prompt,
            [{"role": "user", "content": user_content}],
            max_tokens=budget,
        )
        text = response.choices[0].message.content.strip()
        finish_reason = getattr(response.choices[0], "finish_reason", "stop") or "stop"

        if finish_reason != "length":
            return text
        # Truncated — retry with more tokens

    # Last resort: return what we have (may be slightly cut)
    return text


# ─────────────────────────────────────────────
# 7.  Orchestrator
# ─────────────────────────────────────────────
def run_pipeline(state: VizState, status_container, critic_rounds: int = 3, max_repairs: int = 3):
    """Run full agent pipeline with live status updates.

    Args:
        critic_rounds: Number of plan↔critic refinement loops (1–5).
        max_repairs:   Max auto-repair attempts for broken code (1–5).
    """

    def update(msg, done=False):
        icon = "✅" if done else "⏳"
        status_container.markdown(f"**{icon} {msg}**")

    # — Planner —
    update("Planner crafting analysis strategy...")
    state.plan = planner_agent(state)

    # — Critic loop (up to critic_rounds) —
    for round_num in range(1, critic_rounds + 1):
        update(f"Critic reviewing plan (round {round_num}/{critic_rounds})...")
        approved, feedback = critic_agent(state, round_num)
        state.critic_feedback.append(f"Round {round_num}: {feedback}")

        if approved:
            update(f"Critic approved plan after round {round_num}", done=True)
            break

        if round_num < critic_rounds:
            update(f"Planner revising based on critic feedback (round {round_num})...")
            state.plan = planner_revise_agent(state)

    # — Executor —
    update("Executor generating visualization code...")
    state.fig, state.code, state.error = executor_agent(state)

    # — Repair loop (exec) —
    while state.error and state.retry_exec < max_repairs:
        state.retry_exec += 1
        update(f"Repairing visualization code (attempt {state.retry_exec}/{max_repairs})...")
        try:
            state.fig, state.code, state.error = repair_exec_agent(state)
        except Exception as e:
            state.error = str(e)

    # — Narrative —
    if state.fig:
        update("Narrative agent computing statistics...")
        state.narrative_text, state.narrative_code, state.error = narrative_agent(state)

        # — Repair loop (narrative) —
        while state.error and state.retry_narrative < max_repairs:
            state.retry_narrative += 1
            update(f"Repairing narrative code (attempt {state.retry_narrative}/{max_repairs})...")
            try:
                state.narrative_text, state.narrative_code, state.error = repair_narrative_agent(state)
            except Exception as e:
                state.error = str(e)

    # — Explainer —
    update("Explainer synthesizing final insights...")
    state.explanation = explainer_agent(state)

    # — Serialize fig —
    if state.fig:
        state.fig_png = fig_to_png(state.fig)
        plt.close(state.fig)

    update("Pipeline complete ✨", done=True)
    return state


# ─────────────────────────────────────────────
# 8.  Streamlit UI
# ─────────────────────────────────────────────

# ── Sidebar: pipeline info + settings ──
with st.sidebar:
    st.markdown('<div class="pipeline-header">Agent Pipeline</div>', unsafe_allow_html=True)
    agents = [
        ("🎯", "Clarifier", "Asks follow-up questions"),
        ("📝", "Planner", "Designs analysis strategy"),
        ("🔍", "Critic", "Reviews & refines plan"),
        ("⚙️", "Executor", "Generates & runs code"),
        ("📜", "Narrative", "Computes statistics"),
        ("💡", "Explainer", "Synthesizes insights"),
    ]
    for icon, name, desc in agents:
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-label">{desc}</div>
            <div class="agent-name">{icon} {name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="pipeline-header">Pipeline Settings</div>', unsafe_allow_html=True)
    critic_rounds = st.slider(
        "🔍 Critic feedback loops",
        min_value=1, max_value=5, value=3, step=1,
        help="How many rounds of plan↔critic refinement before proceeding to code generation. More rounds = higher quality plan, slower pipeline."
    )
    max_repairs = st.slider(
        "🔧 Max auto-repair attempts",
        min_value=1, max_value=5, value=3, step=1,
        help="How many times the repair agent retries broken visualization or narrative code before giving up."
    )

# ── Main area ──
# Session state init
for key, default in {
    "conversation_history": [],       # list of dicts with turn data
    "clarifying_questions": [],
    "clarifying_answers": None,
    "awaiting_clarification": False,
    "instructions": "",
    "_run_followup_now": None,        # set to instruction string to trigger inline pipeline run
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Data overview (collapsible, full width) ──
with st.expander("📄 Dataset Overview", expanded=False):
    total_rows = len(df)
    page_size = 10
    total_pages = max(1, math.ceil(total_rows / page_size))
    col_pg, col_info = st.columns([1, 3])
    with col_pg:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="data_page")
    with col_info:
        st.markdown(f"<br><span style='color:#888; font-size:0.85rem;'>Showing rows {((page-1)*page_size)+1}–{min(page*page_size, total_rows)} of {total_rows} · {len(df.columns)} columns</span>", unsafe_allow_html=True)
    st.dataframe(df.iloc[(page-1)*page_size : page*page_size], use_container_width=True)

st.markdown("---")


def _run_analysis(instructions: str, is_followup: bool = False):
    """Run the full agent pipeline and store the result in conversation history."""
    prior_ctx = _build_prior_context(st.session_state.conversation_history) if is_followup else ""

    state = VizState(
        instructions=instructions,
        clarifying_answers=st.session_state.get("clarifying_answers") or "",
        prior_context=prior_ctx,
    )
    with st.status("Running agent pipeline...", expanded=True) as s:
        state = run_pipeline(state, s, critic_rounds=critic_rounds, max_repairs=max_repairs)
        s.update(label="Done!", state="complete")

    # Store turn in conversation history
    st.session_state.conversation_history.append({
        "instructions": instructions,
        "clarifying_answers": state.clarifying_answers,
        "plan": state.plan,
        "code": state.code,
        "narrative_text": state.narrative_text,
        "explanation": state.explanation,
        "fig_png": state.fig_png,
        "critic_feedback": state.critic_feedback,
        "error": state.error,
        "is_followup": is_followup,
    })


def _reset_analysis():
    """Clear all state for a fresh start."""
    st.session_state.conversation_history = []
    st.session_state.clarifying_questions = []
    st.session_state.clarifying_answers = None
    st.session_state.awaiting_clarification = False
    st.session_state.instructions = ""
    st.session_state._run_followup_now = None


def _scroll_to_bottom():
    """Inject JS to scroll the Streamlit app to the bottom after render."""
    st.markdown(
        """
        <script>
            window.addEventListener('load', function() {
                // Small delay to let Streamlit finish rendering
                setTimeout(function() {
                    window.parent.document.querySelector('section.main').scrollTo({
                        top: 999999, behavior: 'smooth'
                    });
                }, 300);
            });
            // Also try immediately in case load already fired
            setTimeout(function() {
                const main = window.parent.document.querySelector('section.main');
                if (main) main.scrollTo({top: 999999, behavior: 'smooth'});
            }, 500);
        </script>
        """,
        unsafe_allow_html=True,
    )


def _render_turn(turn: dict, turn_idx: int):
    """Render a single conversation turn (reusable for history and live results)."""
    # Turn header
    if turn.get("is_followup"):
        st.markdown(f"#### 🔄 Follow-up: {turn['instructions']}")
    else:
        st.markdown(f"#### 📊 Analysis: {turn['instructions']}")

    # Chart
    if turn.get("fig_png"):
        st.image(turn["fig_png"], use_container_width=True)

    # Insight
    if turn.get("explanation"):
        st.markdown("#### 💡 Key Insight")
        st.write(turn["explanation"])

    # Critic feedback expander
    if turn.get("critic_feedback"):
        with st.expander(f"🔍 Critic's review log (turn {turn_idx + 1})", expanded=False):
            for fb in turn["critic_feedback"]:
                st.markdown(f"- {fb}")

    # Plan expander
    if turn.get("plan"):
        with st.expander(f"📝 Analysis plan (turn {turn_idx + 1})", expanded=False):
            st.markdown(turn["plan"])

    # Error notice
    if turn.get("error"):
        st.warning(f"⚠️ Some agents hit errors that couldn't be auto-repaired: {turn['error']}")

    st.markdown("---")


# ═════════════════════════════════════════════
#   CHAT-LIKE UI FLOW
# ═════════════════════════════════════════════

# ── Question input ──

st.markdown("### What would you like to explore?")

_has_history = bool(st.session_state.conversation_history)
_awaiting = st.session_state.get("awaiting_clarification", False)

if not _has_history and not _awaiting:
    # ── Editable input ──
    instructions = st.text_area(
        "Your question or visualization request:",
        placeholder="e.g. Show me the relationship between innovation and agility across industries.",
        height=90,
        label_visibility="collapsed",
    )

    if st.button("▶ Analyse", use_container_width=False):
        if not instructions.strip():
            st.warning("Please enter a question or request before analysing.")
        else:
            st.session_state.instructions = instructions.strip()
            st.session_state.clarifying_answers = None
            st.session_state.clarifying_questions = []

            with st.spinner("Thinking about your question..."):
                questions = clarifier_agent(st.session_state.instructions)

            if questions:
                st.session_state.clarifying_questions = questions
                st.session_state.awaiting_clarification = True
            else:
                _run_analysis(st.session_state.instructions, is_followup=False)
            st.rerun()
else:
    # ── Read-only display of the original question ──
    original_q = ""
    if _has_history:
        original_q = st.session_state.conversation_history[0].get("instructions", "")
    elif st.session_state.instructions:
        original_q = st.session_state.instructions

    if original_q:
        st.markdown(
            f'<div class="agent-card" style="background:#fffdf9;">'
            f'<span style="color:#666; font-size:0.75rem; text-transform:uppercase; '
            f'letter-spacing:0.1em;">Your question</span><br>'
            f'<span style="font-size:1.05rem;">{original_q}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Initial clarifying questions (before first analysis) ──
if (st.session_state.get("awaiting_clarification")
        and st.session_state.get("clarifying_questions")
        and not st.session_state.conversation_history):
    st.markdown("**A couple of quick questions to sharpen the analysis:**")
    answers = {}
    for i, q in enumerate(st.session_state.clarifying_questions):
        answers[i] = st.text_input(q, key=f"clarify_{i}")

    if st.button("✓ Submit & Analyse", use_container_width=False):
        combined_answers = "\n".join(
            f"Q: {st.session_state.clarifying_questions[i]}\nA: {a}"
            for i, a in answers.items() if a.strip()
        )
        st.session_state.clarifying_answers = combined_answers
        st.session_state.awaiting_clarification = False
        # Clear questions and widget keys so they don't ghost-render on rerun
        for i in range(len(st.session_state.clarifying_questions)):
            st.session_state.pop(f"clarify_{i}", None)
        st.session_state.clarifying_questions = []
        _run_analysis(st.session_state.instructions, is_followup=False)
        st.rerun()

# ── Render all past turns ──
for turn_idx, turn in enumerate(st.session_state.conversation_history):
    _render_turn(turn, turn_idx)

# ── After history: follow-up area ──
if st.session_state.conversation_history:

    # Check if the follow-up button was just clicked (flag set by the button handler below).
    # If so, run the pipeline INLINE — the results render right here, below the history,
    # before the next follow-up input box.  No st.rerun() needed.
    if st.session_state.get("_run_followup_now"):
        instr = st.session_state._run_followup_now
        st.session_state._run_followup_now = None

        # Show the question being processed
        st.markdown(f"#### 🔄 Follow-up: {instr}")

        with st.spinner("Thinking about your follow-up..."):
            questions = clarifier_agent(instr, history=st.session_state.conversation_history)

        if questions:
            # Need clarification — save state and rerun so the clarification UI renders
            st.session_state.clarifying_questions = questions
            st.session_state.awaiting_clarification = True
            st.session_state.instructions = instr
            _scroll_to_bottom()
            st.rerun()
        else:
            # No clarification needed — run pipeline inline right here
            st.session_state.instructions = instr
            _run_analysis(instr, is_followup=True)
            # Render the new turn immediately (it's now the last item in history)
            new_turn = st.session_state.conversation_history[-1]
            _render_turn(new_turn, len(st.session_state.conversation_history) - 1)
            _scroll_to_bottom()
            # Fall through to render the follow-up input below

    # ── Follow-up clarification (if awaiting) ──
    if (st.session_state.get("awaiting_clarification")
            and st.session_state.get("clarifying_questions")):

        st.markdown("### Follow-up clarification")
        followup_q = st.session_state.instructions
        st.markdown(
            f'<div class="agent-card" style="background:#fffdf9;">'
            f'<span style="color:#666; font-size:0.75rem; text-transform:uppercase; '
            f'letter-spacing:0.1em;">Your follow-up</span><br>'
            f'<span style="font-size:1.05rem;">{followup_q}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**A couple of quick questions to sharpen the analysis:**")
        answers = {}
        for i, q in enumerate(st.session_state.clarifying_questions):
            answers[i] = st.text_input(q, key=f"followup_clarify_{i}")

        col_submit, col_new, _ = st.columns([1, 1, 3])
        with col_submit:
            if st.button("✓ Submit & Analyse", use_container_width=False, key="followup_submit"):
                combined_answers = "\n".join(
                    f"Q: {st.session_state.clarifying_questions[i]}\nA: {a}"
                    for i, a in answers.items() if a.strip()
                )
                st.session_state.clarifying_answers = combined_answers
                st.session_state.awaiting_clarification = False
                # Clear widget keys
                for i in range(len(st.session_state.clarifying_questions)):
                    st.session_state.pop(f"followup_clarify_{i}", None)
                st.session_state.clarifying_questions = []
                _run_analysis(st.session_state.instructions, is_followup=True)
                _scroll_to_bottom()
                st.rerun()
        with col_new:
            if st.button("✨ New Analysis", use_container_width=False, key="new_from_clarify"):
                _reset_analysis()
                st.rerun()

        _scroll_to_bottom()

    else:
        # ── Normal follow-up input ──
        # Count follow-up turns (exclude the initial analysis)
        followup_count = sum(1 for t in st.session_state.conversation_history if t.get("is_followup"))
        max_followups = 5
        followups_remaining = max_followups - followup_count
        at_limit = followups_remaining <= 0

        st.markdown("### Ask a follow-up question")

        if at_limit:
            st.info(
                f"You've reached the maximum of {max_followups} follow-up questions. "
                "Hit **✨ New Analysis** to start a fresh exploration."
            )
        else:
            st.markdown(
                f"<span style='color:#888; font-size:0.85rem;'>"
                f"The AI will keep all context from the analysis above. "
                f"Ask to refine, drill deeper, compare differently, or explore a new angle. "
                f"({followups_remaining} follow-up{'s' if followups_remaining != 1 else ''} remaining)"
                f"</span>",
                unsafe_allow_html=True,
            )

        followup = st.text_area(
            "Follow-up question:",
            placeholder="e.g. Now break this down by industry. / Show me just the top 5 companies. / How does Sanofi compare to the median?",
            height=75,
            label_visibility="collapsed",
            disabled=at_limit,
            key=f"followup_input_{len(st.session_state.conversation_history)}",
        )

        col_follow, col_new, _ = st.columns([1, 1, 3])
        with col_follow:
            followup_clicked = st.button("🔄 Follow Up", use_container_width=False, disabled=at_limit)
        with col_new:
            new_clicked = st.button("✨ New Analysis", use_container_width=False)

        if followup_clicked:
            if not followup.strip():
                st.warning("Please enter a follow-up question.")
            else:
                # Set a flag so the pipeline runs inline on the NEXT rerun,
                # rendered below the history — not at the top.
                st.session_state._run_followup_now = followup.strip()
                st.session_state.clarifying_answers = None
                st.session_state.clarifying_questions = []
                st.session_state.awaiting_clarification = False
                _scroll_to_bottom()
                st.rerun()

        if new_clicked:
            _reset_analysis()
            st.rerun()
