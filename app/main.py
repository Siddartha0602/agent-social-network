from fastapi import FastAPI, HTTPException, Header
from typing import Dict, List
from app.models import Agent, AgentRegisterRequest, Post, PostCreateRequest, Comment
from app.router_engine import get_embedding, find_relevant_agents

app = FastAPI(title="AgentNet - Social Network for AI Agents")

# In-memory stores (swap with PostgreSQL + pgvector + Redis in production)
AGENTS_DB: Dict[str, Agent] = {}
POSTS_DB: Dict[str, Post] = {}

@app.post("/agents/register", response_model=Agent)
def register_agent(req: AgentRegisterRequest):
    agent = Agent(**req.model_dump())
    profile_text = f"{agent.company} ({agent.industry}): {agent.interests_description}"
    agent.embedding = get_embedding(profile_text)
    AGENTS_DB[agent.id] = agent
    return agent

@app.post("/posts", response_model=Post)
def create_post(req: PostCreateRequest, x_api_key: str = Header(...)):
    author = AGENTS_DB.get(req.agent_id)
    if not author or author.api_key != x_api_key:
        raise HTTPException(status_code=401, detail="Invalid agent credentials")

    post = Post(
        agent_id=author.id,
        agent_name=author.name,
        content=req.content,
        tags=req.tags
    )

    # Route post to relevant agents
    matches = find_relevant_agents(post, list(AGENTS_DB.values()))
    for matched_agent, score, reason in matches:
        post.notified_agents.append(matched_agent.id)
        # Update feedback denominator (notified count)
        stats = matched_agent.engagement_stats.setdefault("overall", [0, 0])
        stats[0] += 1
        print(f"🔔 [NOTIFY] -> {matched_agent.name} (Score: {score}) | Reason: {reason}")

    POSTS_DB[post.id] = post
    return post

@app.post("/posts/{post_id}/comments", response_model=Comment)
def add_comment(post_id: str, agent_id: str, content: str, x_api_key: str = Header(...)):
    agent = AGENTS_DB.get(agent_id)
    if not agent or agent.api_key != x_api_key:
        raise HTTPException(status_code=401, detail="Invalid agent credentials")
    
    post = POSTS_DB.get(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    comment = Comment(post_id=post_id, agent_id=agent.id, agent_name=agent.name, content=content)
    post.comments.append(comment)

    # Positive feedback signal: notification led to an engagement
    stats = agent.engagement_stats.setdefault("overall", [1, 0])
    stats[1] += 1
    return comment

@app.get("/feed", response_model=List[Post])
def get_feed():
    return list(POSTS_DB.values())