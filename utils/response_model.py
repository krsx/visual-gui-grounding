from pydantic import BaseModel, Field
from typing import Literal, Optional


class ActionDetails(BaseModel):
    action_type: Literal["CLICK", "TYPE", "SCROLL", "ANSWER"]
    content: Optional[str] = None
    option_number: int
    option_description: str


class ThoughtResponse(BaseModel):
    thought: str
    action: ActionDetails
