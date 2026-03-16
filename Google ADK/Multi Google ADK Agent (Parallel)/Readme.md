# Event Planning Coordinator with Google ADK (Parallel Orchestration)

A lightweight **multi-agent event planner** built with **Google Agent Development Kit (ADK)** + **Gemini** (free tier).  
It helps plan small-to-medium events (birthdays, family gatherings, team outings in Chennai) by running specialist agents **in parallel**, then combining results into one cohesive plan.

### How It Works

1. **Parallel Phase** (runs simultaneously):
   - **VenueAgent** → Suggests 3–4 realistic Chennai venues with costs & pros/cons
   - **CateringAgent** → Proposes menu, cost per person, veg/non-veg options
   - **ActivitiesAgent** → Recommends games, entertainment, timeline ideas
   - **BudgetTrackerAgent** → Calculates total estimate + savings tips

2. **Coordinator Phase** (after parallel finishes):
   - Merges all outputs into a friendly, markdown-formatted event plan

Uses ADK's `ParallelAgent` + `SequentialAgent` for clean orchestration.

## Features

- Completely **free** (Gemini 1.5 Flash free tier)
- True parallel execution of independent research agents
- Chennai-localized suggestions (venues, food, weather considerations)
- Outputs clean markdown plan with budget breakdown & next steps
- Simple console runner + optional beautiful web UI via `adk web`

## Prerequisites

- Python 3.10+
- Free Gemini API key → https://aistudio.google.com/app/apikey

## Quick Start

```bash
git clone https://github.com/yourusername/event-planner-adk.git
cd event-planner-adk

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install google-adk google-generativeai
