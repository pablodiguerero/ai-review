from pydantic import BaseModel, ConfigDict, Field


class OpenAIResponseUsageSchema(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIInputMessageSchema(BaseModel):
    role: str
    content: str


class OpenAIResponseContentSchema(BaseModel):
    type: str
    text: str | None = None


class OpenAIResponseOutputSchema(BaseModel):
    type: str
    role: str | None = None
    content: list[OpenAIResponseContentSchema] | None = None


class OpenAIResponsesRequestSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: list[OpenAIInputMessageSchema]
    stream: bool = False
    temperature: float | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    text: dict | None = None


class OpenAIResponsesResponseSchema(BaseModel):
    usage: OpenAIResponseUsageSchema = Field(default_factory=OpenAIResponseUsageSchema)
    output: list[OpenAIResponseOutputSchema]
    status: str | None = None
    incomplete_details: dict | None = None

    @property
    def first_text(self) -> str:
        results: list[str] = []
        for block in self.output:
            if block.type == "message" and block.content:
                for content in block.content:
                    if content.type == "output_text" and content.text:
                        results.append(content.text)

        return "".join(results).strip()


class OpenAIResponsesStreamEventSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    delta: str | None = None
    message: str | None = None
    response: dict | None = None
