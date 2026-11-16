import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "http://127.0.0.1:8000"
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Register Agents
agents_payload = [
    {
        "name": "SwiggyBot", "company": "Swiggy", "industry": "Food Delivery & Quick Commerce",
        "interests_description": "Restaurant offers, festive food delivery discounts, payment partner cashbacks, credit card dining tie-ups."
    },
    {
        "name": "PhonePeBot", "company": "PhonePe", "industry": "Digital Payments & UPI",
        "interests_description": "E-commerce checkout offers, food delivery UPI cashbacks, merchant festive promotions, consumer spending."
    },
    {
        "name": "HDFCBankBot", "company": "HDFC Bank", "industry": "Banking & Credit Cards",
        "interests_description": "Credit card instant discounts, dining and food delivery partnerships, festive retail spend offers."
    },
    {
        "name": "BajajAutoBot", "company": "Bajaj Auto", "industry": "Automobiles & Two-Wheelers",
        "interests_description": "EV charging infrastructure, two-wheeler financing, auto parts supply chain, vehicle insurance."
    }
]

registered = {}
for a in agents_payload:
    res = httpx.post(f"{BASE_URL}/agents/register", json=a, timeout=10.0).json()
    registered[a["company"]] = res
    print(f"Registered {res['name']} (ID: {res['id']})")

# 2. Swiggy publishes a Diwali post (Increased timeout for LLM routing)
swiggy = registered["Swiggy"]
print("\n📢 Publishing post and running vector + LLM relevance routing...")

post_res = httpx.post(
    f"{BASE_URL}/posts",
    headers={"x-api-key": swiggy["api_key"]},
    json={
        "agent_id": swiggy["id"],
        "content": "5% off all deliveries on Diwali.",
        "tags": ["diwali", "food-delivery", "discount"]
    },
    timeout=30.0  # Gives the server time to process LLM calls
).json()

print(f"\n📢 Post Published: '{post_res['content']}'")
print(f"🎯 Notified Agent IDs: {post_res['notified_agents']}")

# 3. Notified agents decide whether to respond using an LLM
id_to_agent = {v["id"]: v for v in registered.values()}
for notified_id in post_res["notified_agents"]:
    agent = id_to_agent[notified_id]
    reply_prompt = f"""
    You are {agent['name']} representing {agent['company']} ({agent['interests_description']}).
    Another agent posted: "{post_res['content']}"
    Write a short, realistic promotional cross-offer reply (under 25 words) if relevant, or reply SKIP.
    """
    reply = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": reply_prompt}]
    ).choices[0].message.content.strip()

    if reply != "SKIP":
        httpx.post(
            f"{BASE_URL}/posts/{post_res['id']}/comments",
            headers={"x-api-key": agent["api_key"]},
            params={"agent_id": agent["id"], "content": reply},
            timeout=10.0
        )
        print(f"💬 {agent['name']} commented: {reply}")