from pydantic import BaseModel, Field


class ControlEventIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)


class ControlMessageIn(BaseModel):
    agent_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)


class ControlSpeedIn(BaseModel):
    speed: float = Field(ge=0.1, le=5.0)


class ControlAgentAddIn(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=64)
    traits: str = Field(default="нейтральный, любопытный", min_length=1, max_length=256)
    avatar: str | None = Field(default=None, max_length=64)
    mood: int = Field(default=0, ge=-100, le=100)
    pos_x: float | None = Field(default=None, ge=-20.0, le=20.0)
    pos_z: float | None = Field(default=None, ge=-20.0, le=20.0)


class ControlAgentRemoveIn(BaseModel):
    agent_id: str = Field(min_length=1, max_length=32)
