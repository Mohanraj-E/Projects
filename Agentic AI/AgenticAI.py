# notebook-python
import os
import time
import traceback
import re
from datetime import datetime
from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import schedule

# ── Config ───────────────────────────────────────────────────────────────

GOOGLE_API_KEY = "AIzaSyCsIUiObLaXwkGUKRVRV9iE9gzwg8WFjF8"

# ── Token Budget Guard ───────────────────────────────────────────────────

MAX_OUTPUT_TOKENS = 4096
MIN_OUTPUT_CHARS = 1500   # 🔒 enforce minimum useful digest

# ── LLM Initialization (Retry + Fallback + Hard Fail) ────────────────────

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.35,
        top_p=0.85,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=30,
        max_retries=3,
    )
    print("LLM initialized with gemini-2.5-pro")

except Exception:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.35,
            top_p=0.85,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            timeout=30,
            max_retries=3,
        )
        print("Fallback LLM initialized with gemini-2.5-flash")

    except Exception:
        print("❌ Failed to initialize Gemini LLM:")
        traceback.print_exc()
        raise

# ── Tool: Forced Web Search ──────────────────────────────────────────────

search = DuckDuckGoSearchRun()

@tool
def search_ai_news(query: str) -> list:
    """Search latest AI news and return raw results."""
    return search.invoke(f"{query} after:2026-01-15")

# ── LangGraph State ──────────────────────────────────────────────────────

class DigestState(TypedDict):
    date: str
    queries: List[str]
    raw_results: list
    sources: list
    digest: str

# ── Planner Node ─────────────────────────────────────────────────────────

def planner_node(state: DigestState):
    return {
        "queries": [
            "latest Generative AI releases",
            "new LLM models research",
            "RAG retrieval augmented generation updates",
            "AI agents frameworks",
            "multimodal AI tools launch",
            "enterprise AI platforms news",
            "open source AI tools",
            "AI safety and governance",
        ]
    }

# ── Search Node (FORCED TOOLS) ───────────────────────────────────────────

def search_node(state: DigestState):
    normalized = []

    for q in state["queries"]:
        try:
            res = search_ai_news.invoke(q)

            # Case 1: list of dicts
            if isinstance(res, list):
                for r in res:
                    if isinstance(r, dict):
                        normalized.append(r)
                    elif isinstance(r, str):
                        normalized.append({
                            "title": q,
                            "url": "",
                            "snippet": r
                        })

            # Case 2: single string
            elif isinstance(res, str):
                normalized.append({
                    "title": q,
                    "url": "",
                    "snippet": res
                })

        except Exception:
            continue

    return {"raw_results": normalized}


# ── Filter + Deduplicate ─────────────────────────────────────────────────

def filter_node(state: DigestState):
    seen = set()
    sources = []

    for r in state["raw_results"]:
        if not isinstance(r, dict):
            continue

        url = r.get("link") or r.get("url") or ""
        if url.startswith("http") and url not in seen:
            seen.add(url)
            sources.append({
                "title": r.get("title", "AI Update"),
                "url": url,
                "snippet": r.get("snippet", "")
            })

    return {"sources": sources[:15]}


# ── Summarization Node (STRICT SOURCE-ONLY) ──────────────────────────────

def summarize_node(state: DigestState):
    today = state["date"]

    context = "\n\n".join(
        f"Title: {s['title']}\nURL: {s['url']}\nSnippet: {s['snippet']}"
        for s in state["sources"]
    )

    prompt = f"""
You are an expert AI newsletter curator.

STRICT RULES:
- Use ONLY the sources below
- Minimum 10 topics
- 2–4 sentence summary each
- Include "Why it matters"
- Include real Source URLs INLINE
- NO hallucination
- NO tool mentions
- Output MUST exceed {MIN_OUTPUT_CHARS} characters

FORMAT EXACTLY:

Daily AI Digest – {today}

1. Title
   Summary...
   Why it matters: ...
   Source: https://...

SOURCES:
{context}
"""

    result = llm.invoke([HumanMessage(content=prompt)]).content

    if len(result) < MIN_OUTPUT_CHARS:
        raise ValueError("Digest too short – token budget guard triggered")

    return {"digest": result.strip()}

# ── Build LangGraph ──────────────────────────────────────────────────────

graph = StateGraph(DigestState)

graph.add_node("planner", planner_node)
graph.add_node("search", search_node)
graph.add_node("filter", filter_node)
graph.add_node("summarize", summarize_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "search")
graph.add_edge("search", "filter")
graph.add_edge("filter", "summarize")
graph.add_edge("summarize", END)

digest_app = graph.compile()

# ── Digest Generator ─────────────────────────────────────────────────────

def generate_daily_digest() -> str:
    print("Starting generate_daily_digest...")
    try:
        state = digest_app.invoke({
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        return state["digest"]

    except Exception:
        print("Digest generation failed:")
        traceback.print_exc()
        raise

# ── Email Formatting (UNCHANGED) ─────────────────────────────────────────

def format_digest_for_email(raw_digest: str) -> str:
    cleaned = raw_digest.strip()

    html = f"""
<html><body style="font-family:Arial;line-height:1.7;max-width:760px;margin:auto">
<h2>{cleaned.splitlines()[0]}</h2>
"""

    for line in cleaned.splitlines()[1:]:
        if re.match(r'^\d+\.', line):
            html += f"<h3>{line}</h3>"
        elif line.startswith("Why it matters"):
            html += f"<p><b>{line}</b></p>"
        elif line.startswith("Source"):
            url = line.split(":",1)[1].strip()
            html += f'<p>Source: <a href="{url}">{url}</a></p>'
        else:
            html += f"<p>{line}</p>"

    html += "</body></html>"
    return html

# ── Email Sender ─────────────────────────────────────────────────────────

def send_email(html_content: str):
    SENDER_EMAIL = "horrorfiction7@gmail.com"
    SENDER_PASSWORD = "kxuzmquzqcvdmmof"
    RECIPIENT = "mohanrajezhilarasu@gmail.com"

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT
    msg["Subject"] = f"Daily AI Digest – {datetime.now().strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(html_content, "html"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()

# ── Daily Job ────────────────────────────────────────────────────────────

def daily_job():
    print("Running daily job...")
    digest = generate_daily_digest()
    html = format_digest_for_email(digest)
    send_email(html)
    print("✅ Digest sent successfully")

# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    daily_job()
