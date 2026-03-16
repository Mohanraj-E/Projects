
# Daily AI Digest – Automated Newsletter with Gemini + LangGraph

**Automated daily email newsletter** that curates the latest AI news, model releases, RAG/agent/multimodal developments, and more — powered by **Google Gemini 2.5** and **LangGraph**.

Sends a clean, human-like digest to your inbox every day (or on schedule), summarizing 10+ high-quality topics with "Why it matters" insights and direct source links — **no hallucinations**, strict source-only generation.

Built as a lightweight, production-ready Python script — ideal for personal use, learning agentic workflows, or extending into a full SaaS newsletter.

## ✨ Features

- **Daily AI news curation** — covers GenAI releases, LLMs, RAG, agents, multimodal tools, open-source, safety & governance
- **Forced web search** with date filtering (`after:2026-01-15`) via DuckDuckGo
- **LangGraph workflow** — clean state machine: Planner → Search → Filter → Summarize
- **Gemini 2.5 Pro / Flash** fallback + retry logic
- **Token & length guard** — enforces minimum useful output (~1500+ chars)
- **HTML email formatting** — readable, clickable sources, mobile-friendly
- **Scheduled execution** ready (using `schedule` library — extendable to cron / cloud functions)

## 🖼️ Example Output (in email)

Daily AI Digest – 2026-03-16  
1. New Gemini 2.5 Pro Update  
   Google released enhanced reasoning capabilities...  
   Why it matters: Better multi-step planning for agents...  
   Source: https://blog.google/technology/ai/...

...and 10–15 more curated items.

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5-pro (fallback: 2.5-flash) via `langchain-google-genai`
- **Agent Framework**: LangGraph (stateful graph) + LangChain tools
- **Search**: DuckDuckGoSearchRun (no API key needed)
- **Email**: `smtplib` + Gmail SMTP (easy to swap with SendGrid / Resend)
- **Scheduling**: `schedule` library (simple in-script cron-like)

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Gmail account with **App Password** enabled (not regular password!)
