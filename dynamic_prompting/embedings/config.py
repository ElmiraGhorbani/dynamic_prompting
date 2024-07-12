from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingsConfig(BaseModel):
    model_name: Literal["mxbai-embed-large-v1", "bge-small-en-v1.5",
                        "nomic-embed-text-v1.5"] = Field(default="nomic-embed-text-v1.5")
    local_files: bool = Field(default=True)
    trust_remote_code: bool = Field(default=False)
    max_dimension: int = Field(default=None)
    epsilon: float = Field(default=1e-5)

    class Config:
        protected_namespaces = ()
