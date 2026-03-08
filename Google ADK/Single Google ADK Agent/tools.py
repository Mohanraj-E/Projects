import pandas as pd
from io import StringIO

def parse_expenses(expenses_text: str) -> dict:
    """
    Parse expense list or CSV-like text into categories and totals.
    Example input: 
    Food,1200
    Rent,18000
    Transport,1500
    Entertainment,800
    """
    try:
        df = pd.read_csv(StringIO(expenses_text), header=None, names=["category", "amount"])
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna()
        
        total = df["amount"].sum()
        by_category = df.groupby("category")["amount"].sum().to_dict()
        
        return {
            "total_expenses": total,
            "breakdown": by_category
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_savings_rate(income: float, total_expenses: float) -> float:
    """Calculate savings rate as percentage"""
    if income <= 0:
        return 0.0
    savings = income - total_expenses
    rate = (savings / income) * 100
    return round(rate, 1)

def suggest_budget(income: float, expenses_breakdown: dict, savings_goal: float = 20.0) -> str:
    """Generate simple budget advice"""
    total_exp = sum(expenses_breakdown.values())
    current_rate = calculate_savings_rate(income, total_exp)
    
    advice = [f"Current savings rate: {current_rate}% (goal was {savings_goal}%)"]
    
    if current_rate < savings_goal:
        shortfall = savings_goal - current_rate
        advice.append(f"You need to save ~{shortfall:.1f}% more per month.")
        
        # Suggest cuts on highest non-essential (heuristic)
        sorted_exp = sorted(expenses_breakdown.items(), key=lambda x: x[1], reverse=True)
        for cat, amt in sorted_exp[:3]:  # top 3 categories
            if cat.lower() in ["food", "dining", "entertainment", "transport", "shopping"]:
                cut = round(amt * 0.25, 0)
                advice.append(f"→ Try reducing '{cat}' by ₹{cut}/month → potential +{round((cut/income)*100,1)}% savings")
    
    if "rent" in expenses_breakdown and expenses_breakdown["rent"] > income * 0.4:
        advice.append("→ Rent >40% of income — consider long-term housing optimization.")
    
    return "\n".join(advice)