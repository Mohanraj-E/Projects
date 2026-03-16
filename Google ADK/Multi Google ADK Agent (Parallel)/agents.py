import os
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

# ── Gemini Setup (free tier model) ──
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = Gemini(
    model="gemini-1.5-flash",          # or "gemini-2.0-flash" if available
    generation_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 1000,
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
)

# ── 1. Venue Agent ──
venue_agent = LlmAgent(
    name="VenueAgent",
    model=MODEL,
    description="Suggests suitable venues in Chennai area with cost & logistics.",
    instruction="""
You are a Chennai-local event venue expert.
Given event details: {event_description?}, guest_count: {guest_count?}, budget_level: {budget_level?}, date: {event_date?}

Suggest 3–4 realistic venues (banquet halls, rooftop, beachside, community halls, etc.).
For each: name, approx cost (₹), capacity, pros/cons, booking tips, rain backup if outdoor.

Output ONLY JSON:
{{
  "venues": [
    {{"name": "...", "cost_range": "₹20,000–35,000", "capacity": "50–120", "pros": "...", "cons": "..."}}
  ]
}}
""",
    output_key="venue_suggestions"
)

# ── 2. Catering Agent ──
catering_agent = LlmAgent(
    name="CateringAgent",
    model=MODEL,
    description="Plans menu & catering costs.",
    instruction="""
You are a catering specialist for Chennai events.
Use: {event_description?}, {guest_count?}, dietary needs if mentioned.

Suggest:
- Menu (starters, main, dessert, beverages)
- Approx cost per person
- Veg/non-veg options, local favorites (idli, biryani, etc.)

Output ONLY JSON:
{{
  "menu_summary": "...",
  "cost_per_person": "₹450–700",
  "total_estimate": "₹...",
  "notes": "..."
}}
""",
    output_key="catering_plan"
)

# ── 3. Activities Agent ──
activities_agent = LlmAgent(
    name="ActivitiesAgent",
    model=MODEL,
    description="Recommends fun activities & entertainment.",
    instruction="""
You suggest engaging activities for the event type: {event_description?}, group size: {guest_count?}.

Include: icebreakers, games, music/DJ, photo booth, etc.
Estimate time & any extra cost.

Output ONLY JSON:
{{
  "activities": ["Activity 1 - desc - ₹cost", ...],
  "timeline_suggestion": "..."
}}
""",
    output_key="activities_plan"
)

# ── 4. Budget Tracker Agent ──
budget_agent = LlmAgent(
    name="BudgetTrackerAgent",
    model=MODEL,
    description="Estimates total budget & savings tips.",
    instruction="""
You are a budget-conscious planner.
Use venue, catering, activities outputs + overall budget_level: {budget_level?}

Calculate rough total cost.
Suggest savings ideas (DIY, off-peak booking, etc.).

Output ONLY JSON:
{{
  "total_estimated_cost": "₹80,000–1,20,000",
  "breakdown": {{ "venue": "...", "catering": "...", ... }},
  "savings_tips": ["tip1", "tip2"]
}}
""",
    output_key="budget_summary"
)

# ── Parallel Fan-Out (runs the 4 specialists concurrently) ──
parallel_team = ParallelAgent(
    name="EventSpecialists",
    sub_agents=[venue_agent, catering_agent, activities_agent, budget_agent],
    description="Runs venue, catering, activities & budget research in parallel"
)

# ── Final Coordinator (gathers & writes nice plan) ──
coordinator = LlmAgent(
    name="EventCoordinator",
    model=MODEL,
    description="Merges all parallel results into a beautiful event plan.",
    instruction="""
You are the lead event planner.
Combine:
- Venues: {venue_suggestions?}
- Catering: {catering_plan?}
- Activities: {activities_plan?}
- Budget: {budget_summary?}

Create a friendly, complete event plan in markdown:
# Event Plan: [short title]

## Overview
...

## Recommended Venue
...

## Menu & Catering
...

## Activities & Schedule
...

## Budget Breakdown & Tips
...

End with next steps or questions to refine.
Use Chennai-friendly tone, realistic for 2026.
""",
    output_key="final_event_plan"
)

# ── Root Workflow: Sequential (first parallel → then coordinator) ──
event_planner = SequentialAgent(
    name="EventPlanningCoordinator",
    sub_agents=[parallel_team, coordinator],
    description="Parallel research → final coordinated event plan"
)