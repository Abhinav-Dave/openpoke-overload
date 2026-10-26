from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


class RootResponse(BaseModel):
    status: str
    service: str
    version: str
    endpoints: List[str]


class SetTimezoneRequest(BaseModel):
    timezone: str


class SetTimezoneResponse(BaseModel):
    ok: bool = True
    timezone: str


class LoadResponse(BaseModel):
    agents: Dict[str, int]
    usage: Dict[str, Any]
