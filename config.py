from pydantic import BaseModel


class Config(BaseModel):
    USE_TRAJECTORY: bool = False
    USE_EXTERNAL_DATA: bool = True
