from pydantic import BaseModel


class Config(BaseModel):
    USE_TRAJECTORY: bool = False
