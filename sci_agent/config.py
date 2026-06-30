from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class AgentConfig:
    temperature: float = 0.7
    max_tokens: int = 4096
    max_steps: int = 50
    max_retries: int = 3

    memory_dir: str = "./memory_store"
    working_memory_max_messages: int = 240
    working_memory_max_tokens: int = 96000

    complexity_level: Optional[str] = None
    seed: Optional[int] = None

    verbose: bool = True
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    @classmethod
    def from_file(cls, path: str) -> AgentConfig:
        with open(path) as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load .yaml config files.")
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                try:
                    import yaml
                    yaml.dump(self.to_dict(), f, default_flow_style=False)
                except ImportError:
                    json.dump(self.to_dict(), f, indent=2)
            else:
                json.dump(self.to_dict(), f, indent=2)
