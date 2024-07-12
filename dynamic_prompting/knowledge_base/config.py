from pydantic import BaseModel, Field


class KnowledgeBaseConfig(BaseModel):
    knowledge_base_name: str = Field(default="signal_classification")
