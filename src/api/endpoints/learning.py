# src/api/endpoints/learning.py
from fastapi import APIRouter
from ...features.learning.agent import initialize_learning_agent

learning_agent_executor = initialize_learning_agent()
router = APIRouter()
