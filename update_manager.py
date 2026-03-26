from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from crawl import DEFAULT_LIMIT, DEFAULT_OUTPUT_DIR, DEFAULT_SORT_BY, build_model_record, fetch_ranked_models, save_model_outputs
from filter_from_csv import build_graph_dataset

DATA_DIR = Path(DEFAULT_OUTPUT_DIR)
CONFIG_PATH = DATA_DIR / "update_config.json"
STATUS_PATH = DATA_DIR / "update_status.json"
OUTPUT_JSON_PATH = DATA_DIR / "hf_models_ultimate.json"

TIMEZONE_NAME = "Asia/Shanghai"
MAX_LIMIT = 1000

DEFAULT_CONFIG = {
    "crawl_limit": DEFAULT_LIMIT,
    "sort_by": DEFAULT_SORT_BY,
    "weekly_interval_days": 7,
    "last_successful_update": None,
}

DEFAULT_STATUS = {
    "state": "idle",
    "message": "",
    "current_limit": DEFAULT_LIMIT,
    "last_trigger": None,
    "last_started_at": None,
    "last_completed_at": None,
    "next_scheduled_update_at": None,
    "last_diff": {
        "added_count": 0,
        "removed_count": 0,
        "unchanged_count": 0,
        "added_models": [],
        "removed_models": [],
    },
    "last_error": "",
}


def now_local() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE_NAME))


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return copy.deepcopy(default)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_data_dir()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config() -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(_load_json(CONFIG_PATH, DEFAULT_CONFIG))
    config["crawl_limit"] = validate_limit(config.get("crawl_limit", DEFAULT_LIMIT))
    config["weekly_interval_days"] = max(1, int(config.get("weekly_interval_days", 7)))
    config["sort_by"] = str(config.get("sort_by", DEFAULT_SORT_BY) or DEFAULT_SORT_BY)
    return config


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = load_config()
    normalized.update(config)
    normalized["crawl_limit"] = validate_limit(normalized.get("crawl_limit", DEFAULT_LIMIT))
    normalized["weekly_interval_days"] = max(1, int(normalized.get("weekly_interval_days", 7)))
    normalized["sort_by"] = str(normalized.get("sort_by", DEFAULT_SORT_BY) or DEFAULT_SORT_BY)
    _save_json(CONFIG_PATH, normalized)
    return normalized


def load_status() -> dict[str, Any]:
    status = copy.deepcopy(DEFAULT_STATUS)
    status.update(_load_json(STATUS_PATH, DEFAULT_STATUS))
    return status


def save_status(status: dict[str, Any]) -> dict[str, Any]:
    merged = load_status()
    merged.update(status)
    _save_json(STATUS_PATH, merged)
    return merged


def validate_limit(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = DEFAULT_LIMIT
    return min(MAX_LIMIT, max(1, value))


def load_existing_records() -> list[dict[str, Any]]:
    if not OUTPUT_JSON_PATH.exists():
        return []
    try:
        payload = json.loads(OUTPUT_JSON_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _compute_next_scheduled_at(
    last_successful_update: str | None,
    weekly_interval_days: int,
) -> str | None:
    if not last_successful_update:
        return None
    try:
        last_dt = datetime.fromisoformat(last_successful_update)
    except ValueError:
        return None
    return (last_dt + timedelta(days=weekly_interval_days)).isoformat()


def is_update_due(config: dict[str, Any], status: dict[str, Any] | None = None) -> bool:
    effective_status = status or load_status()
    if effective_status.get("state") == "running":
        return False

    last_successful = config.get("last_successful_update")
    if not last_successful:
        return False

    next_run = _compute_next_scheduled_at(last_successful, int(config.get("weekly_interval_days", 7)))
    if not next_run:
        return False
    try:
        return now_local() >= datetime.fromisoformat(next_run)
    except ValueError:
        return False


def get_dataset_stats() -> dict[str, int]:
    records = load_existing_records()
    if not records:
        return {"models": 0, "relations": 0, "organizations": 0}

    model_count = len(records)
    relation_count = 0
    organizations = set()
    for record in records:
        model_id = str(record.get("model_id", "") or "")
        base_model = str(record.get("base_model_final", "") or "")
        if base_model and base_model not in {"", "基础模型", "未知"} and base_model != model_id:
            relation_count += 1
        author = str(record.get("author", "") or "").strip()
        if author and author != "未知":
            organizations.add(author)

    return {
        "models": model_count,
        "relations": relation_count,
        "organizations": len(organizations),
    }


def get_status_payload() -> dict[str, Any]:
    config = load_config()
    status = load_status()
    status["next_scheduled_update_at"] = _compute_next_scheduled_at(
        config.get("last_successful_update"),
        int(config.get("weekly_interval_days", 7)),
    )
    return {
        "config": config,
        "status": status,
        "stats": get_dataset_stats(),
    }


def mark_update_started(limit: int, trigger: str) -> dict[str, Any]:
    config = load_config()
    next_limit = validate_limit(limit)
    if config.get("crawl_limit") != next_limit:
        config["crawl_limit"] = next_limit
        save_config(config)

    started_at = now_local().isoformat()
    return save_status(
        {
            "state": "running",
            "message": "正在更新数据，请稍候。",
            "current_limit": next_limit,
            "last_trigger": trigger,
            "last_started_at": started_at,
            "last_error": "",
        }
    )


def mark_update_failed(limit: int, trigger: str, error_message: str) -> dict[str, Any]:
    config = load_config()
    return save_status(
        {
            "state": "failed",
            "message": "更新失败，请查看错误信息。",
            "current_limit": validate_limit(limit),
            "last_trigger": trigger,
            "last_completed_at": now_local().isoformat(),
            "next_scheduled_update_at": _compute_next_scheduled_at(
                config.get("last_successful_update"),
                int(config.get("weekly_interval_days", 7)),
            ),
            "last_error": error_message,
        }
    )


def run_incremental_update(limit: int | None = None, trigger: str = "manual") -> dict[str, Any]:
    config = load_config()
    effective_limit = validate_limit(limit if limit is not None else config.get("crawl_limit", DEFAULT_LIMIT))
    sort_by = str(config.get("sort_by", DEFAULT_SORT_BY) or DEFAULT_SORT_BY)
    started_at = now_local().isoformat()

    models = fetch_ranked_models(limit=effective_limit, sort_by=sort_by)
    current_order = [getattr(model, "modelId", "") for model in models]
    current_order = [model_id for model_id in current_order if model_id]

    existing_records = load_existing_records()
    existing_map = {str(record.get("model_id", "")): record for record in existing_records if record.get("model_id")}
    current_set = set(current_order)
    existing_set = set(existing_map.keys())

    added_ids = [model_id for model_id in current_order if model_id not in existing_set]
    removed_ids = sorted(existing_set - current_set)
    unchanged_ids = [model_id for model_id in current_order if model_id in existing_set]

    fresh_models_by_id = {getattr(model, "modelId", ""): model for model in models}
    fresh_records: dict[str, dict[str, Any]] = {}
    for model_id in added_ids:
        fresh_records[model_id] = build_model_record(fresh_models_by_id[model_id])

    merged_records: list[dict[str, Any]] = []
    for rank, model_id in enumerate(current_order, start=1):
        if model_id in fresh_records:
            record = dict(fresh_records[model_id])
        else:
            record = dict(existing_map[model_id])
        record["rank"] = rank
        record["top_n_limit"] = effective_limit
        merged_records.append(record)

    output_summary = save_model_outputs(merged_records, output_dir=DATA_DIR)
    graph_summary = build_graph_dataset()

    completed_at = now_local().isoformat()
    config["crawl_limit"] = effective_limit
    config["last_successful_update"] = completed_at
    save_config(config)

    status = save_status(
        {
            "state": "idle",
            "message": "更新完成。",
            "current_limit": effective_limit,
            "last_trigger": trigger,
            "last_started_at": started_at,
            "last_completed_at": completed_at,
            "next_scheduled_update_at": _compute_next_scheduled_at(
                completed_at,
                int(config.get("weekly_interval_days", 7)),
            ),
            "last_diff": {
                "added_count": len(added_ids),
                "removed_count": len(removed_ids),
                "unchanged_count": len(unchanged_ids),
                "added_models": added_ids,
                "removed_models": removed_ids,
            },
            "last_error": "",
        }
    )

    return {
        "config": config,
        "status": status,
        "stats": get_dataset_stats(),
        "summary": {
            "output": output_summary,
            "graph": graph_summary,
            "added_models": added_ids,
            "removed_models": removed_ids,
            "unchanged_models": unchanged_ids,
        },
    }
