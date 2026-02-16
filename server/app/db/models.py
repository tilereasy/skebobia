from pydantic import BaseModel, Field


class ControlEventIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)


class ControlMessageIn(BaseModel):
    agent_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)


class ControlSpeedIn(BaseModel):
    speed: float = Field(ge=0.1, le=5.0)
