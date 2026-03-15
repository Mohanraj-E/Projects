
# Sequential Content Creation Workflow with Google ADK

A simple **multi-agent sequential pipeline** built with **Google Agent Development Kit (ADK)** and **Gemini** (free tier) that turns any blog topic into a polished, SEO-ready markdown article.

Agents run one after another:

1. **Ideator** → Brainstorms title + key points  
2. **Outliner** → Builds detailed section structure  
3. **Drafter** → Writes full engaging draft (~600–900 words)  
4. **Optimizer** → Polishes grammar, adds SEO, CTA, and image suggestions

All agents share context via ADK session state — zero external APIs beyond Gemini.

## Features

- Completely **free** to run (Gemini 1.5 Flash free tier)  
- Sequential orchestration using `SequentialAgent`  
- Friendly, conversational Indian English style  
- Outputs clean markdown with suggested images  
- Easy to extend (add research tool, change tone, etc.)  
- Local console runner + optional nice web UI via `adk web`

## Prerequisites

- Python 3.10+
- Gemini API key (free): https://aistudio.google.com/app/apikey

## Installation

```bash
git clone https://github.com/yourusername/content-workflow-adk.git
cd content-workflow-adk

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install google-adk google-generativeai
