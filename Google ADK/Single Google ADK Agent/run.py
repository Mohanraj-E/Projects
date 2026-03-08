"""
run.py - Simple async console runner for Personal Finance Advisor
"""
import asyncio
from agent import finance_agent
from google.adk.runners import InMemoryRunner

async def run_conversation(runner):
    print("Personal Finance Advisor (ADK + Gemini) - type 'exit' to quit\n")
    print("Tell me about your income, expenses, goals...\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Stay financially strong 💪")
            break

        if not user_input:
            continue

        print("\nAdvisor is thinking...\n")

        # Await the async method
        response = await runner.run_debug(user_input)

        # Safe text extraction
        if hasattr(response, "text"):
            reply = response.text.strip()
        elif hasattr(response, "content") and response.content.parts:
            reply = response.content.parts[0].text.strip()
        else:
            reply = str(response).strip()

        print("Advisor:", reply)
        print("-" * 80)

def main():
    runner = InMemoryRunner(
        agent=finance_agent,
        app_name="PersonalFinanceConsoleApp"
    )

    # Run the async conversation loop
    asyncio.run(run_conversation(runner))

if __name__ == "__main__":
    import os

    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY not found.")
        print("In PowerShell:")
        print("   $env:GEMINI_API_KEY = 'AIzaSyCsIUiObLaXwkGUKRVRV9iE9gzwg8WFjF8'")
        print("Then run: python run.py")
        exit(1)

    main()