"""Configuração da API via variáveis de ambiente / .env (nunca commitadas)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    root: Path
    database_url: str | None
    experiment_config: Path
    artifacts_root: Path
    thesis_root: Path
    cors_origins: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = root or Path(__file__).resolve().parents[4]
        _load_dotenv(root)
        return cls(
            root=root,
            database_url=os.environ.get("DATABASE_URL"),
            experiment_config=Path(
                os.environ.get("FLOWBUILDER_CONFIG", root / "experiments/e0/config.json")
            ),
            artifacts_root=Path(
                os.environ.get("FLOWBUILDER_ARTIFACTS", root / "experiments/api_runs")
            ),
            thesis_root=Path(
                os.environ.get("FALCO_THESIS_ROOT", root.parent / "tesedaru")
            ),
            cors_origins=[
                o.strip()
                for o in os.environ.get(
                    "FLOWBUILDER_CORS", "http://localhost:5173"
                ).split(",")
                if o.strip()
            ],
        )
