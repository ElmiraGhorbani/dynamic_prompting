from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    rank: int = Field(default=0)
    world_size: int = Field(default=1)
    max_seq_len: int = Field(default=1024)
    max_batch_size: int = Field(default=8)
    local_files: bool = Field(default=True)

class PromptConfig(BaseModel):
    prompt : str = Field(default="")