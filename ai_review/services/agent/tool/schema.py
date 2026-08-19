from pydantic import BaseModel


class AgentToolResultSchema(BaseModel):
    command: str
    output: str
    executed: bool
