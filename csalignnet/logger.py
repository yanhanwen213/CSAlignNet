"""Epoch CSV/JSONL training logger (logs under <log_root>/)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class EpochLogger:
    def __init__(self, log_root: str | Path, run_name: str | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"pida_kfold_{stamp}"
        self.dir = Path(log_root) / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.dir / "metrics.csv"
        self.jsonl_path = self.dir / "metrics.jsonl"
        self.txt_path = self.dir / "train.log"
        self.config_path = self.dir / "config.json"
        self._csv_fields: list[str] | None = None
        self._rows: list[dict[str, Any]] = []

    def save_config(self, config: dict) -> None:
        payload = {
            "run_name": self.run_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **config,
        }
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def log_epoch(self, epoch: int, metrics: dict[str, Any], message: str | None = None) -> None:
        row = {"epoch": int(epoch), **metrics}
        self._rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        fields = list(row.keys())
        if self._csv_fields is None:
            self._csv_fields = fields
        else:
            for k in fields:
                if k not in self._csv_fields:
                    self._csv_fields.append(k)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in self._rows:
                writer.writerow(r)

        line = message or (
            f"Epoch {epoch:03d} | " + " ".join(f"{k}={v}" for k, v in metrics.items() if k != "epoch")
        )
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def path(self) -> Path:
        return self.dir
