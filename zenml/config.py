from pydantic import BaseModel


class Config(BaseModel):
    USE_TRAJECTORY: bool = False
    USE_MANUAL_PHASE: bool = False  # Need to use trajectory if want to use this.
    USE_EXTERNAL_DATA: bool = True
    USE_ALL_FEATURES: bool = False
    NUM_FEATURES: int = 15
    N_TRIALS: int = 50
    TIMEOUT: int = 600
    OPTIMIZATION_MODEL: str = "XGBoost"
