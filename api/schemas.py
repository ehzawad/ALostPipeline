from pydantic import BaseModel, Field

class RequestBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    messages: str = Field("[]", description="JSON string of message history")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "example question",
                "messages": "[]"
            }
        }
