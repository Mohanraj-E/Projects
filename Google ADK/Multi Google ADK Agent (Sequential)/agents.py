import os
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

# ── Gemini Setup (free model) ──
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = Gemini(
    model="gemini-1.5-flash-latest",          # or "gemini-2.0-flash" if available in 2026 free tier
    generation_config={
        "temperature": 0.75,
        "top_p": 0.92,
        "max_output_tokens": 1200,
    },
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
)

# ── 1. Ideator Agent ──
ideator = LlmAgent(
    name="Ideator",
    model=MODEL,
    description="Brainstorms creative blog ideas, titles and key points.",
    instruction="""
You are a creative blog ideator. Given a topic, generate:
- 1 catchy, SEO-friendly title (50-60 chars)
- 5-7 bullet points as main ideas / sections

Be engaging, modern, valuable. Target Indian / global English readers.
Output ONLY JSON:
{
  "title": "...",
  "key_points": ["point 1", "point 2", ...]
}
""",
    output_key="blog_ideas"   # Saves result to session.state["blog_ideas"]
)

# ── 2. Outliner Agent ──
outliner = LlmAgent(
    name="Outliner",
    model=MODEL,
    description="Creates detailed structured outline from ideas.",
    instruction="""
You are a professional blog outliner.
Use the ideas from previous step ({blog_ideas?}).

Create a clear structure:
- Introduction (hook + thesis)
- 4-6 main sections with 3-5 sub-bullets each
- Conclusion (summary + CTA)

Output ONLY JSON:
{
  "title": "...",           // improved if needed
  "outline": {
    "introduction": "...",
    "sections": [
      {"heading": "...", "subpoints": ["...", ...]},
      ...
    ],
    "conclusion": "..."
  }
}
Keep language natural and reader-friendly.
""",
    output_key="blog_outline"
)

# ── 3. Drafter Agent ──
drafter = LlmAgent(
    name="Drafter",
    model=MODEL,
    description="Writes full, fluent blog post draft.",
    instruction="""
You are an engaging blog writer.
Write a complete 600-900 word article using this outline: {blog_outline?}

Style: friendly, conversational, Indian English, helpful tone.
Use short paragraphs, subheadings, bold key phrases.
Add 1-2 real-world examples or tips.
End with CTA (subscribe, comment, share).

Output the FULL markdown article.
""",
    output_key="blog_draft"
)

# ── 4. Optimizer Agent ──
optimizer = LlmAgent(
    name="Optimizer",
    model=MODEL,
    description="Polishes draft, adds SEO & image suggestions.",
    instruction="""
You are a senior content editor + SEO specialist.
Improve this draft: {blog_draft?}

Tasks:
- Fix grammar, flow, repetition
- Add 3-5 SEO keywords naturally (primary + secondary)
- Strengthen intro hook & conclusion CTA
- Suggest 2-3 stock image ideas (describe briefly)
- Keep length similar

Output in markdown with these sections at end:

## Final Optimized Post
[full improved article]

## Suggested Images
- Image 1: description
- Image 2: ...
""",
    output_key="final_post"
)

# ── Sequential Workflow (the pipeline) ──
content_pipeline = SequentialAgent(
    name="ContentCreationWorkflow",
    sub_agents=[ideator, outliner, drafter, optimizer],
    description="Sequential blog writing pipeline: idea → outline → draft → polish"
)