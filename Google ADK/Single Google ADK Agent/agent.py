"""
agent.py - Personal Finance Advisor using Google ADK + Gemini (new SDK 2026)
"""
import os

# ── Google ADK ──
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# ── Your custom tools ──
from tools import suggest_budget, calculate_savings_rate, parse_expenses

# ── API key check ──
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError(
        "GEMINI_API_KEY environment variable not found.\n"
        "In PowerShell: $env:GEMINI_API_KEY = 'your-key-here'\n"
        "Then run python run.py again."
    )

# ── Tools ──
tools = [
    FunctionTool(parse_expenses),
    FunctionTool(calculate_savings_rate),
    FunctionTool(suggest_budget),
]

# ── Finance Advisor Agent ──
# Pass model name as string — ADK creates the underlying model/client
finance_agent = LlmAgent(
    name="PersonalFinanceAdvisor",
    model="gemini-1.5-flash-latest",          # ← this is the key change
    description=(
        "A practical, encouraging personal finance assistant for people in India. "
        "Works in Indian Rupees (₹). Friendly, realistic, non-judgmental tone."
    ),
    instruction="""
You are a helpful personal finance advisor for people in India.

Your goals:
- Help users understand their current financial situation
- Assist with budgeting and saving
- Give realistic, actionable advice
- Be encouraging and non-judgmental

Always:
- Use ₹ symbol for money amounts
- Speak in friendly, conversational Indian English
- Ask clarifying questions when information is missing 
  (monthly take-home income, fixed/variable expenses, financial goals, debts, etc.)

Typical flow:
1. If user shares expenses (list, text, CSV-like), use parse_expenses tool
2. Calculate savings rate with calculate_savings_rate when income & expenses are known
3. Provide budget suggestions using suggest_budget tool
4. Give general tips when appropriate:
   - Build emergency fund (3–6 months expenses)
   - Pay high-interest debt first
   - Start small investments (mutual funds, RD, SIPs)
   - Track expenses regularly

Keep responses clear, short-to-medium length, and motivating.
""",
    tools=tools,
    
    # Optional: if ADK supports passing generation/safety config separately, add here
    # generation_config={"temperature": 0.7, "max_output_tokens": 1024},
    # Otherwise defaults are usually fine for starters
)