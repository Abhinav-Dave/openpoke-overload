from __future__ import annotations

from ...config import DATA_DIR
from .models import TriggerRecord
from .service import TriggerService
from .store import TriggerStore


_default_db_path = DATA_DIR / "triggers.db"
_trigger_store = TriggerStore(_default_db_path)
_trigger_service = TriggerService(_trigger_store)


def get_trigger_service() -> TriggerService:
    return _trigger_service


__all__ = [
    "TriggerRecord",
    "TriggerService",
    "get_trigger_service",
]
