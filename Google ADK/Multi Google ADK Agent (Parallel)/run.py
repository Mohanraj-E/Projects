import asyncio
import os
from agents import event_planner
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

def main():
    print("Event Planning Coordinator (ADK + Gemini - Parallel Orchestration)")
    print("Describe your event (e.g. '30th birthday party for 40 people, mid-budget, Saturday evening in Chennai')")
    print("You can also add structured info: guest_count=40, budget_level=mid, event_date=2026-04-10")
    print("Type 'exit' to quit\n")

    # ── Use InMemoryRunner (LocalRunner is not available) ──
    runner = InMemoryRunner(
        agent=event_planner,                        # your root/parallel orchestrator agent
        app_name="EventPlanningParallelDemo"        # required in recent ADK versions
    )

    # Session service for maintaining state across agents if needed
    session_service = InMemorySessionService()

    while True:
        user_input = input("Event details: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Happy planning! 🎉")
            break
        if not user_input:
            continue

        print("\nPlanning in parallel... (usually 40–90 seconds)\n")

        try:
            # Most common pattern in recent ADK: run_debug takes prompt as positional arg
            # If your version expects messages or different signature → adjust accordingly
            response = asyncio.run(
                runner.run_debug(user_input)  # ← positional string input
                # Alternative patterns (uncomment one if above fails):
                # runner.run_debug(messages=[{"role": "user", "parts": [user_input]}])
                # runner.run(user_input)
            )

            # Extract final text safely
            if hasattr(response, 'text'):
                final_plan = response.text.strip()
            elif hasattr(response, 'content') and response.content.parts:
                final_plan = response.content.parts[0].text.strip()
            else:
                final_plan = str(response).strip()

            print("=" * 80)
            print("YOUR EVENT PLAN\n")
            print(final_plan)
            print("=" * 80)

        except TypeError as e:
            print("Runner call error:", str(e))
            print("Try adjusting the run_debug / run call signature.")
            print("Common alternatives:")
            print("  - runner.run_debug(user_input)")
            print("  - runner.run(user_input)")
            print("  - runner.run_debug(messages=[{'role': 'user', 'parts': [user_input]}])")
            break  # or continue depending on your preference

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Set GEMINI_API_KEY environment variable.")
        print("In PowerShell:")
        print("   $env:GEMINI_API_KEY = 'AIzaSyCsIUiObLaXwkGUKRVRV9iE9gzwg8WFjF8'")
        exit(1)

    main()