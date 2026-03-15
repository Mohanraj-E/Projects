import asyncio
import os
from agents import content_pipeline
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

def main():
    print("Sequential Content Creation Workflow (ADK + Gemini)")
    print("Enter a blog topic (e.g. 'Best Budget Smartphones Under ₹25000 in 2026')")
    print("Type 'exit' to quit\n")

    runner = InMemoryRunner(
        agent=content_pipeline,
        app_name="SequentialContentCreation"
    )

    session_service = InMemorySessionService()  # optional

    while True:
        topic = input("Topic: ").strip()
        if topic.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Keep creating great content ✍️")
            break
        if not topic:
            continue

        print("\nPipeline running... (this may take 30–90 seconds)\n")

        # Try positional argument first
        response = asyncio.run(runner.run_debug(topic))

        # Fallback: if above fails, try one of these alternatives:
        # response = asyncio.run(runner.run(topic))
        # response = asyncio.run(runner.run_debug(messages=[{"role": "user", "parts": [topic]}]))

        if hasattr(response, 'text'):
            final_content = response.text.strip()
        else:
            final_content = str(response).strip()

        print("=" * 70)
        print("FINAL BLOG POST\n")
        print(final_content)
        print("=" * 70)

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Set GEMINI_API_KEY environment variable.")
        print("In PowerShell: $env:GEMINI_API_KEY = 'AIza...'")
        exit(1)
    
    main()