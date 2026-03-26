from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from crawl import MULTIMODAL_COMPONENTS, NO_BASE_MODEL

DEFAULT_INPUT_CSV = Path("data/hf_models_valid.csv")
DEFAULT_OUTPUT_CSV = Path("hf_models_has_base.csv")


def _stringify_value(value: Any) -> str:
    if value is None:
        return "未知"
    text = str(value).strip()
    return text if text else "未知"


def build_graph_dataset(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> dict[str, int]:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        empty_df = pd.DataFrame()
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return {"total_rows": 0, "rows_with_base": 0, "missing_base_rows": 0, "final_rows": 0}

    df = pd.read_csv(input_path)
    if df.empty:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return {"total_rows": 0, "rows_with_base": 0, "missing_base_rows": 0, "final_rows": 0}

    df_has_base = df[
        (df["base_model_final"] != NO_BASE_MODEL)
        & (df["base_model_final"] != "获取失败")
        & (df["base_model_final"] != MULTIMODAL_COMPONENTS)
    ].copy()

    base_model_ids = df_has_base["base_model_final"].dropna().astype(str).unique().tolist()
    df_exist_bases = df[df["model_id"].isin(base_model_ids)].copy()
    missing_bases = [model_id for model_id in base_model_ids if model_id not in df["model_id"].values]

    missing_rows = []
    for model_id in missing_bases:
        missing_rows.append(
            {
                "model_id": model_id,
                "author": "未知",
                "downloads": "未知",
                "likes": "未知",
                "last_modified": "未知",
                "library_name": "未知",
                "pipeline_tag": "未知",
                "tags": "未知",
                "l1_card": "未知",
                "l2_main_config": "未知",
                "l3_separate_config": "未知",
                "l4_diffusers_config": "未知",
                "l5_readme": "未知",
                "l6_tags": "未知",
                "l7_model_id": "未知",
                "l8_model_type": "未知",
                "l9_safetensors_index": "未知",
                "base_model_final": NO_BASE_MODEL,
                "is_base_model_valid": "未知",
                "is_multimodal": "未知",
                "dataset_deps": "未知",
                "extract_source": "未知",
                "status": "success",
                "error_msg": "",
            }
        )

    df_missing_bases = pd.DataFrame(missing_rows)
    df_final = pd.concat([df_has_base, df_exist_bases, df_missing_bases], ignore_index=True)
    if not df_final.empty:
        df_final = df_final.drop_duplicates(subset=["model_id"], keep="first")

        if "dataset_deps" in df_final.columns:
            df_final["dataset_deps"] = df_final["dataset_deps"].apply(_stringify_value)
        if "is_base_model_valid" in df_final.columns:
            df_final["is_base_model_valid"] = df_final["is_base_model_valid"].apply(_stringify_value)

    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "total_rows": len(df),
        "rows_with_base": len(df_has_base),
        "missing_base_rows": len(missing_rows),
        "final_rows": len(df_final),
    }


if __name__ == "__main__":
    print(json.dumps(build_graph_dataset(), ensure_ascii=False, indent=2))
