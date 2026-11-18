"""HTTP API around the same prediction code the dispatch desk uses.

Run:  uvicorn api:app --port 8000      Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import predict
from common import CHOICES

app = FastAPI(title="Call Severity Triage API", version="2.0")


class Call(BaseModel):
    """The five required fields, plus any optional detail fields the model understands."""
    road_type: str
    speed_limit: int | str
    weather: str
    light_condition: str
    junction_detail: str
    optional: dict[str, str] = Field(default_factory=dict, description="Optional extra fields, e.g. road_surface_conditions")

    def as_call(self) -> dict:
        return {**self.model_dump(exclude={"optional"}), **self.optional}


class WhatIf(BaseModel):
    call: Call
    changes: dict[str, str | int]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/choices")
def choices() -> dict:
    """Valid values for each required field."""
    return CHOICES


@app.post("/predict")
def predict_endpoint(call: Call) -> dict:
    try:
        return predict.predict_call(call.as_call())
    except predict.ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/what-if")
def what_if_endpoint(body: WhatIf) -> dict:
    try:
        result = predict.what_if(body.call.as_call(), dict(body.changes))
    except predict.ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))
    if not result:
        raise HTTPException(status_code=503, detail="The v2 model is not available on this server")
    return result
