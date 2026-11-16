import os
import json
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from app.models import Agent, Post

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> List[float]:
    return embedder.encode(text, normalize_embeddings=True).tolist()

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    return float(np.dot(np.array(vec_a), np.array(vec_b)))

def find_relevant_agents(post: Post, all_agents: List[Agent], top_k: int = 5) -> List[Tuple[Agent, float, str]]:
    post_text = f"{post.content} Tags: {', '.join(post.tags)}"
    post_vec = get_embedding(post_text)

    candidates = []
    for agent in all_agents:
        if agent.id == post.agent_id or not agent.embedding:
            continue
        sim = cosine_similarity(post_vec, agent.embedding)
        candidates.append((agent, sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    shortlist = candidates[:top_k]

    approved_agents = []
    for agent, sim_score in shortlist:
        effective_threshold = 0.35  # Permissive threshold for interview demo
        
        prompt = f"""
        You are a relevance routing engine for a B2B AI Agent social network.
        Post Author: {post.agent_name}
        Post Content: "{post.content}"
        
        Candidate Agent: {agent.name} ({agent.company} - {agent.industry})
        Candidate Interests: "{agent.interests_description}"
        
        Does this post present a commercial, payment, checkout, or promotional opportunity for {agent.company}?
        You must reply strictly with a JSON object containing "score" (float between 0.0 and 1.0) and "reason" (string). Do not include markdown code blocks.
        Example: {{"score": 0.85, "reason": "Payment tie-in for holiday offer"}}
        """
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80
            )
            raw_content = res.choices[0].message.content.strip()
            # Strip markdown block wrappers if model accidentally adds them
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
                
            data = json.loads(raw_content.strip())
            llm_score = float(data.get("score", 0.0))
            reason = data.get("reason", "")

            final_score = (0.2 * sim_score) + (0.8 * llm_score)
            print(f"👉 Evaluated {agent.name}: Final Score {final_score:.2f} (Threshold: {effective_threshold}) | Reason: {reason}")

            if final_score >= effective_threshold:
                approved_agents.append((agent, round(final_score, 2), reason))
        except Exception as e:
            print(f"⚠️ Routing error for {agent.name}: {e}")

    return approved_agents