from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_model(path: str | Path, model_cls: Type[T]) -> T:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return model_cls.model_validate(data)


def save_model(path: str | Path, model: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump(mode="json", exclude_none=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
