from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from huggingface_hub import HfApi, utils
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_LIMIT = 5
DEFAULT_SORT_BY = "downloads"
RETRY_TIMES = 3
TIMEOUT = 10
SLEEP_TIME = 0.01
VERIFY_BASE_MODEL = True
DEFAULT_OUTPUT_DIR = "data"

UNKNOWN_VALUE = "未知"
NO_BASE_MODEL = "基础模型"
MULTIMODAL_COMPONENTS = "多模态内部组件"

OFFICIAL_BASE_MODEL_MAP = {
    "qwen": "Qwen/Qwen-7B",
    "qwen2": "Qwen/Qwen2-7B",
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen3": "Qwen/Qwen3-7B",
    "qwen3.5": "Qwen/Qwen3.5-27B",
    "llama": "meta-llama/Llama-2-7b-hf",
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.2": "meta-llama/Llama-3.2-8B",
    "llama3.3": "meta-llama/Llama-3.3-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "mistral-nemo": "mistralai/Mistral-Nemo-12B-Instruct",
    "gemma": "google/gemma-7b",
    "gemma2": "google/gemma-2-9b",
    "phi": "microsoft/phi-2",
    "phi2": "microsoft/phi-2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "deepseek": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-v2": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "yi": "01-ai/Yi-6B",
    "baichuan": "baichuan-inc/Baichuan2-7B-Base",
    "internlm": "internlm/internlm2-7b",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "albert": "albert-base-v2",
    "t5": "t5-base",
    "flan-t5": "google/flan-t5-base",
    "bart": "facebook/bart-base",
    "gpt2": "gpt2",
    "vit": "google/vit-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "stable-diffusion": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "yolov": "ultralytics/yolov8x",
    "whisper": "openai/whisper-small",
    "wav2vec2": "facebook/wav2vec2-base",
}


def init_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=RETRY_TIMES,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


SESSION = init_session()
API = HfApi()


def _normalize_string(value: Any, fallback: str = UNKNOWN_VALUE) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, "", UNKNOWN_VALUE):
        return []
    return [str(value).strip()]


def is_missing_base_model(value: Any) -> bool:
    text = _normalize_string(value, "")
    return text in {"", UNKNOWN_VALUE, NO_BASE_MODEL, MULTIMODAL_COMPONENTS}


def verify_model_exists(model_id: str) -> bool:
    if not model_id or is_missing_base_model(model_id):
        return False
    try:
        API.model_info(model_id, timeout=5)
        return True
    except (utils.RepositoryNotFoundError, utils.HfHubHTTPError, Exception):
        return False


def extract_base_from_card(card_data: dict[str, Any]) -> str:
    if not card_data:
        return UNKNOWN_VALUE
    base_model = card_data.get("base_model", UNKNOWN_VALUE)
    if isinstance(base_model, list) and base_model:
        return _normalize_string(base_model[0])
    if isinstance(base_model, str) and base_model.strip():
        return base_model.strip()
    return UNKNOWN_VALUE


def extract_base_from_main_config(model_id: str) -> tuple[str, bool]:
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = SESSION.get(url, timeout=TIMEOUT)
        if response.status_code != 200:
            return UNKNOWN_VALUE, False

        config = response.json()
        candidate_fields = [
            "parent_model_name",
            "base_model_name_or_path",
            "base_model",
            "pretrained_model_name_or_path",
            "_name_or_path",
        ]
        for field in candidate_fields:
            value = config.get(field)
            if value and isinstance(value, str) and len(value.strip()) > 3 and "/" in value:
                return value.strip(), False

        for nested_key in ["lora_config", "adapter_config"]:
            nested_config = config.get(nested_key)
            if isinstance(nested_config, dict):
                value = nested_config.get("base_model_name_or_path")
                if value and isinstance(value, str) and len(value.strip()) > 3:
                    return value.strip(), False

        is_multimodal = "text_config" in config or "vision_config" in config
        return UNKNOWN_VALUE, is_multimodal
    except Exception:
        return UNKNOWN_VALUE, False


def extract_base_from_separate_config(model_id: str) -> str:
    for file_name in ["adapter_config.json", "peft_config.json", "lora_config.json"]:
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/{file_name}"
            response = SESSION.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                config = response.json()
                value = config.get("base_model_name_or_path")
                if value and isinstance(value, str) and len(value.strip()) > 3:
                    return value.strip()
        except Exception:
            continue
    return UNKNOWN_VALUE


def extract_base_from_diffusers_config(model_id: str) -> str:
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/model_index.json"
        response = SESSION.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            config = response.json()
            components = list(config.get("components", {}).keys())
            if components:
                return f"{MULTIMODAL_COMPONENTS}: {','.join(components)}"
    except Exception:
        return UNKNOWN_VALUE
    return UNKNOWN_VALUE


def extract_base_from_readme(model_id: str) -> str:
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        response = SESSION.get(url, timeout=TIMEOUT)
        if response.status_code != 200:
            return UNKNOWN_VALUE

        text = response.text.lower()
        patterns = [
            r"based\s+on\s+([\w\-\/\.]+)",
            r"fine-tuned\s+from\s+([\w\-\/\.]+)",
            r"derived\s+from\s+([\w\-\/\.]+)",
            r"built\s+on\s+top\s+of\s+([\w\-\/\.]+)",
            r"trained\s+from\s+([\w\-\/\.]+)",
            r"initialized\s+from\s+([\w\-\/\.]+)",
            r"forked\s+from\s+([\w\-\/\.]+)",
            r"base\s+model\s*[:=]\s*([\w\-\/\.]+)",
            r"parent\s+model\s*[:=]\s*([\w\-\/\.]+)",
            r"基于\s*([\w\-\/\.]+)\s*(微调|训练|开发)",
            r"基础模型\s*[:：]\s*([\w\-\/\.]+)",
            r"父模型\s*[:：]\s*([\w\-\/\.]+)",
            r"以\s*([\w\-\/\.]+)\s*为(基础|基座)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 3 and "/" in candidate:
                    return candidate
        return UNKNOWN_VALUE
    except Exception:
        return UNKNOWN_VALUE


def extract_base_from_model_tags(tags: Iterable[str] | None) -> str:
    if not tags:
        return UNKNOWN_VALUE
    for tag in tags:
        if "base_model:" in tag:
            return tag.split("base_model:", maxsplit=1)[-1].strip()
    return UNKNOWN_VALUE


def extract_base_from_model_id(model_id: str) -> str:
    model_id_lower = model_id.lower().replace("_", "-").replace(" ", "")
    for family, base_model in OFFICIAL_BASE_MODEL_MAP.items():
        if family in model_id_lower:
            return base_model

    regex_rules = [
        (r"llama[-_]?(\d+)[-_]?(\d+)[bB]", lambda m: f"meta-llama/Llama-{m.group(1)}-{m.group(2)}B-hf"),
        (r"qwen[-_]?(\d+[.\d]*)[-_]?(\d+)[bB]", lambda m: f"Qwen/Qwen{m.group(1)}-{m.group(2)}B"),
        (r"mistral[-_]?(\d+)[bB]", lambda m: f"mistralai/Mistral-{m.group(1)}B-v0.1"),
        (r"yi[-_]?(\d+)[bB]", lambda m: f"01-ai/Yi-{m.group(1)}B"),
    ]
    for pattern, builder in regex_rules:
        match = re.search(pattern, model_id_lower)
        if match:
            try:
                return builder(match)
            except Exception:
                continue
    return UNKNOWN_VALUE


def extract_base_from_model_type(model_type: str) -> str:
    if not model_type:
        return UNKNOWN_VALUE
    return OFFICIAL_BASE_MODEL_MAP.get(model_type.lower(), UNKNOWN_VALUE)


def extract_base_from_safetensors_index(model_id: str) -> str:
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/model.safetensors.index.json"
        response = SESSION.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            config = response.json()
            metadata = config.get("metadata", {})
            value = metadata.get("base_model")
            if value and isinstance(value, str) and len(value.strip()) > 3:
                return value.strip()
    except Exception:
        return UNKNOWN_VALUE
    return UNKNOWN_VALUE


def fetch_ranked_models(limit: int = DEFAULT_LIMIT, sort_by: str = DEFAULT_SORT_BY) -> list[Any]:
    return list(API.list_models(limit=limit, sort=sort_by))


def build_model_record(model: Any, verify_base_model: bool = VERIFY_BASE_MODEL) -> dict[str, Any]:
    model_id = getattr(model, "modelId", "")
    result: dict[str, Any] = {
        "model_id": model_id,
        "author": _normalize_string(getattr(model, "author", None)),
        "downloads": getattr(model, "downloads", 0) or 0,
        "likes": getattr(model, "likes", 0) or 0,
        "last_modified": _normalize_string(getattr(model, "lastModified", None)),
        "library_name": _normalize_string(getattr(model, "library_name", None)),
        "pipeline_tag": _normalize_string(getattr(model, "pipeline_tag", None)),
        "tags": ",".join(getattr(model, "tags", []) or []),
        "l1_card": UNKNOWN_VALUE,
        "l2_main_config": UNKNOWN_VALUE,
        "l3_separate_config": UNKNOWN_VALUE,
        "l4_diffusers_config": UNKNOWN_VALUE,
        "l5_readme": UNKNOWN_VALUE,
        "l6_tags": UNKNOWN_VALUE,
        "l7_model_id": UNKNOWN_VALUE,
        "l8_model_type": UNKNOWN_VALUE,
        "l9_safetensors_index": UNKNOWN_VALUE,
        "base_model_final": NO_BASE_MODEL,
        "is_base_model_valid": False,
        "is_multimodal": False,
        "dataset_deps": [],
        "extract_source": UNKNOWN_VALUE,
        "status": "success",
        "error_msg": "",
    }

    try:
        model_info = API.model_info(model_id, timeout=TIMEOUT)
        card_data = model_info.cardData or {}
        model_config = model_info.config or {}
        model_type = model_config.get("model_type", "")

        result["l1_card"] = extract_base_from_card(card_data)
        result["l2_main_config"], result["is_multimodal"] = extract_base_from_main_config(model_id)
        result["l3_separate_config"] = extract_base_from_separate_config(model_id)
        result["l4_diffusers_config"] = extract_base_from_diffusers_config(model_id)
        result["l5_readme"] = extract_base_from_readme(model_id)
        result["l6_tags"] = extract_base_from_model_tags(getattr(model, "tags", None))
        result["l7_model_id"] = extract_base_from_model_id(model_id)
        result["l8_model_type"] = extract_base_from_model_type(model_type)
        result["l9_safetensors_index"] = extract_base_from_safetensors_index(model_id)
        result["dataset_deps"] = _normalize_list(card_data.get("datasets", []))

        extract_priority = [
            ("l1_card", "模型卡显式标注"),
            ("l2_main_config", "主配置文件"),
            ("l3_separate_config", "独立 Adapter 配置"),
            ("l9_safetensors_index", "Safetensors 索引"),
            ("l5_readme", "README 文档"),
            ("l6_tags", "模型标签"),
            ("l4_diffusers_config", "Diffusers 组件配置"),
            ("l7_model_id", "模型 ID 语义分析"),
            ("l8_model_type", "模型类型映射"),
        ]

        final_base = None
        final_source = None
        for key, source_name in extract_priority:
            candidate = result[key]
            if not is_missing_base_model(candidate):
                final_base = candidate
                final_source = source_name
                break

        if final_base is None and result["is_multimodal"]:
            final_base = MULTIMODAL_COMPONENTS
            final_source = "多模态配置识别"

        result["base_model_final"] = final_base if final_base else NO_BASE_MODEL
        result["extract_source"] = final_source if final_source else "官方基座模型"

        if verify_base_model:
            result["is_base_model_valid"] = verify_model_exists(result["base_model_final"])
    except Exception as exc:
        result["status"] = "failed"
        result["error_msg"] = str(exc)[:200]

    return result


def crawl_models_from_list(
    models: Iterable[Any],
    verify_base_model: bool = VERIFY_BASE_MODEL,
    sleep_time: float = SLEEP_TIME,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    models_list = list(models)
    total = len(models_list)
    for index, model in enumerate(models_list, start=1):
        model_id = getattr(model, "modelId", "")
        print(f"[{index}/{total}] 正在处理: {model_id}")
        record = build_model_record(model, verify_base_model=verify_base_model)
        if record["status"] == "success":
            print(
                f"  提取完成 | 父模型: {record['base_model_final']} | 来源: {record['extract_source']}"
            )
        else:
            print(f"  处理失败 | 原因: {record['error_msg']}")
        records.append(record)
        time.sleep(sleep_time)
    return records


def save_model_outputs(
    model_data: list[dict[str, Any]],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_models = pd.DataFrame(model_data)
    if df_models.empty:
        df_models = pd.DataFrame(
            columns=[
                "model_id",
                "author",
                "downloads",
                "likes",
                "last_modified",
                "library_name",
                "pipeline_tag",
                "tags",
                "l1_card",
                "l2_main_config",
                "l3_separate_config",
                "l4_diffusers_config",
                "l5_readme",
                "l6_tags",
                "l7_model_id",
                "l8_model_type",
                "l9_safetensors_index",
                "base_model_final",
                "is_base_model_valid",
                "is_multimodal",
                "dataset_deps",
                "extract_source",
                "status",
                "error_msg",
            ]
        )

    df_models.to_csv(output_path / "hf_models_ultimate.csv", index=False, encoding="utf-8-sig")
    df_models.to_json(
        output_path / "hf_models_ultimate.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    df_valid = df_models[df_models["status"] == "success"].copy()
    df_valid.to_csv(output_path / "hf_models_valid.csv", index=False, encoding="utf-8-sig")

    valid_with_base = 0
    if not df_valid.empty:
        valid_with_base = int(
            (~df_valid["base_model_final"].apply(is_missing_base_model)).sum()
        )

    return {
        "total_models": len(df_models),
        "success_count": int((df_models["status"] == "success").sum()) if not df_models.empty else 0,
        "failed_count": int((df_models["status"] != "success").sum()) if not df_models.empty else 0,
        "valid_with_base_count": valid_with_base,
    }


def crawl_models(
    limit: int = DEFAULT_LIMIT,
    sort_by: str = DEFAULT_SORT_BY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    verify_base_model: bool = VERIFY_BASE_MODEL,
) -> dict[str, Any]:
    print(f"开始抓取前 {limit} 个模型，排序方式: {sort_by}")
    models = fetch_ranked_models(limit=limit, sort_by=sort_by)
    records = crawl_models_from_list(models, verify_base_model=verify_base_model)
    summary = save_model_outputs(records, output_dir=output_dir)
    summary["limit"] = limit
    summary["sort_by"] = sort_by
    return summary


if __name__ == "__main__":
    result = crawl_models()
    print(json.dumps(result, ensure_ascii=False, indent=2))
