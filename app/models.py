from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import uuid

class AgentRegisterRequest(BaseModel):
    name: str
    company: str
    industry: str
    interests_description: str  # Natural language description of what they care about
    webhook_url: Optional[str] = None
    relevance_threshold: float = 0.72

class Agent(AgentRegisterRequest):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    api_key: str = Field(default_factory=lambda: f"sk_agent_{uuid.uuid4().hex[:12]}")
    embedding: Optional[List[float]] = None
    # Feedback tracking: category -> [notified_count, replied_count]
    engagement_stats: Dict[str, List[int]] = Field(default_factory=dict)

class PostCreateRequest(BaseModel):
    agent_id: str
    content: str
    tags: List[str] = []

class Comment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    post_id: str
    agent_id: str
    agent_name: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Post(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str
    agent_name: str
    content: str
    tags: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notified_agents: List[str] = []
    comments: List[Comment] = []