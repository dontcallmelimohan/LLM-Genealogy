from huggingface_hub import HfApi, utils
import pandas as pd
import json
import time
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ======================================
# ⚙️ 全局配置（按需修改）
# ======================================
LIMIT = 5  # 抓取模型数量
SORT_BY = "downloads"  # 排序规则: downloads/likes/last_modified
RETRY_TIMES = 3  # 网络请求重试次数
TIMEOUT = 10  # 单次请求超时时间(秒)
SLEEP_TIME = 0.01  # 每次请求间隔(防限流)
VERIFY_BASE_MODEL = True  # 是否校验父模型ID真实有效
OUTPUT_DIR = "data"  # 输出目录

# ======================================
# 🎯 全量模型系列-官方基座映射表（持续更新）
# ======================================
OFFICIAL_BASE_MODEL_MAP = {
    # 大语言模型主流系列
    "qwen": "Qwen/Qwen-7B",
    "qwen2": "Qwen/Qwen2-7B",
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen3": "Qwen/Qwen3-7B",
    "qwen3.5": "qwen/Qwen3.5-27B",
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
    # 传统NLP系列
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "albert": "albert-base-v2",
    "t5": "t5-base",
    "flan-t5": "google/flan-t5-base",
    "bart": "facebook/bart-base",
    "gpt2": "gpt2",
    # 多模态/视觉/扩散系列
    "vit": "google/vit-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "stable-diffusion": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "yolov": "ultralytics/yolov8x",
    "whisper": "openai/whisper-small",
    "wav2vec2": "facebook/wav2vec2-base",
}

# ======================================
# 🛠️ 工具函数初始化
# ======================================
# 全局请求会话（连接复用+重试机制，防限流+提升稳定性）
def init_session():
    session = requests.Session()
    retry = Retry(
        total=RETRY_TIMES,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

session = init_session()
api = HfApi()

# ======================================
# 🔍 12层提取核心函数（全覆盖）
# ======================================
def verify_model_exists(model_id: str) -> bool:
    """校验模型ID是否真实存在于Hugging Face"""
    if not model_id or model_id in ["无", "基座模型", "多模态内部组件", ""]:
        return False
    try:
        api.model_info(model_id, timeout=5)
        return True
    except (utils.RepositoryNotFoundError, Exception):
        return False

def extract_base_from_card(card_data: dict) -> str:
    """L1: 从Model Card提取（最高优先级，作者显式标注）"""
    if not card_data:
        return "无"
    # 兼容单值/列表格式
    base_model = card_data.get("base_model", "无")
    if isinstance(base_model, list) and len(base_model) > 0:
        return base_model[0].strip()
    elif isinstance(base_model, str) and base_model.strip() != "":
        return base_model.strip()
    return "无"

def extract_base_from_main_config(model_id: str) -> tuple[str, bool]:
    """L2: 从主config.json提取（核心可靠来源）"""
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return "无", False
        config = resp.json()

        # 全量可能的父模型字段
        candidate_fields = [
            "parent_model_name",
            "base_model_name_or_path",
            "base_model",
            "pretrained_model_name_or_path",
            "_name_or_path"
        ]
        for field in candidate_fields:
            val = config.get(field)
            if val and isinstance(val, str) and len(val.strip()) > 3 and "/" in val:
                return val.strip(), False

        # LoRA/PEFT内嵌配置
        if "lora_config" in config and isinstance(config["lora_config"], dict):
            val = config["lora_config"].get("base_model_name_or_path")
            if val and isinstance(val, str) and len(val.strip()) > 3:
                return val.strip(), False
        if "adapter_config" in config and isinstance(config["adapter_config"], dict):
            val = config["adapter_config"].get("base_model_name_or_path")
            if val and isinstance(val, str) and len(val.strip()) > 3:
                return val.strip(), False

        # 多模态模型标记
        is_multimodal = "text_config" in config or "vision_config" in config
        return "无", is_multimodal
    except Exception:
        return "无", False

def extract_base_from_separate_config(model_id: str) -> str:
    """L3: 从独立的adapter/peft配置文件提取（LoRA模型专属）"""
    config_files = [
        "adapter_config.json",
        "peft_config.json",
        "lora_config.json",
    ]
    for file_name in config_files:
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/{file_name}"
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                config = resp.json()
                val = config.get("base_model_name_or_path")
                if val and isinstance(val, str) and len(val.strip()) > 3:
                    return val.strip()
        except Exception:
            continue
    return "无"

def extract_base_from_diffusers_config(model_id: str) -> str:
    """L4: 从diffusers的model_index.json提取（扩散模型专属）"""
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/model_index.json"
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code == 200:
            config = resp.json()
            components = list(config.get("components", {}).keys())
            if len(components) > 0:
                return f"多组件扩散模型: {','.join(components)}"
    except Exception:
        return "无"
    return "无"

def extract_base_from_readme(model_id: str) -> str:
    """L5: 从README.md全文提取（覆盖作者未显式标注的场景）"""
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return "无"
        text = resp.text.lower()

        # 中英文全量匹配规则
        patterns = [
            # 英文规则
            r"based\s+on\s+([\w\-\/\.]+)",
            r"fine-tuned\s+from\s+([\w\-\/\.]+)",
            r"derived\s+from\s+([\w\-\/\.]+)",
            r"built\s+on\s+top\s+of\s+([\w\-\/\.]+)",
            r"trained\s+from\s+([\w\-\/\.]+)",
            r"initialized\s+from\s+([\w\-\/\.]+)",
            r"forked\s+from\s+([\w\-\/\.]+)",
            r"base\s+model\s*[:=]\s*([\w\-\/\.]+)",
            r"parent\s+model\s*[:=]\s*([\w\-\/\.]+)",
            # 中文规则
            r"基于\s*([\w\-\/\.]+)\s*(微调|训练|开发)",
            r"基座模型\s*[:：]\s*([\w\-\/\.]+)",
            r"父模型\s*[:：]\s*([\w\-\/\.]+)",
            r"以\s*([\w\-\/\.]+)\s*为(基础|基座)",
        ]

        for pat in patterns:
            match = re.search(pat, text)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 3 and "/" in candidate:
                    return candidate
        return "无"
    except Exception:
        return "无"

def extract_base_from_model_tags(tags: list) -> str:
    """L6: 从模型tags提取"""
    if not tags:
        return "无"
    for tag in tags:
        if "base_model:" in tag:
            return tag.split("base_model:")[-1].strip()
    return "无"

def extract_base_from_model_id(model_id: str) -> str:
    """L7: 从模型ID语义解析（兜底核心能力）"""
    model_id_lower = model_id.lower().replace("_", "-").replace(" ", "")
    # 优先匹配已知系列
    for family, base_model in OFFICIAL_BASE_MODEL_MAP.items():
        if family in model_id_lower:
            return base_model
    # 正则精准匹配
    regex_rules = [
        (r"llama[-_]?(\d+)[-_]?(\d+)[bB]", lambda m: f"meta-llama/Llama-{m.group(1)}-{m.group(2)}B-hf"),
        (r"qwen[-_]?(\d+[.\d]*)[-_]?(\d+)[bB]", lambda m: f"Qwen/Qwen{m.group(1)}-{m.group(2)}B"),
        (r"mistral[-_]?(\d+)[bB]", lambda m: f"mistralai/Mistral-{m.group(1)}B-v0.1"),
        (r"yi[-_]?(\d+)[bB]", lambda m: f"01-ai/Yi-{m.group(1)}B"),
    ]
    for pat, builder in regex_rules:
        match = re.search(pat, model_id_lower)
        if match:
            try:
                return builder(match)
            except Exception:
                continue
    return "无"

def extract_base_from_model_type(model_type: str) -> str:
    """L8: 从model_type匹配官方基座"""
    if not model_type:
        return "无"
    return OFFICIAL_BASE_MODEL_MAP.get(model_type.lower(), "无")

def extract_base_from_safetensors_index(model_id: str) -> str:
    """L9: 从safetensors索引文件提取线索"""
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/model.safetensors.index.json"
        resp = session.get(url, timeout=TIMEOUT)
        if resp.status_code == 200:
            config = resp.json()
            metadata = config.get("metadata", {})
            val = metadata.get("base_model")
            if val and isinstance(val, str) and len(val.strip()) > 3:
                return val.strip()
    except Exception:
        return "无"
    return "无"

# ======================================
# 🚀 主执行逻辑
# ======================================
if __name__ == "__main__":
    # 1. 创建输出目录
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 获取模型列表
    print(f"📥 开始获取{LIMIT}个模型，排序规则：{SORT_BY}")
    models = list(api.list_models(limit=LIMIT, sort=SORT_BY))
    model_data = []
    success_count = 0
    failed_count = 0

    # 3. 批量处理每个模型
    for idx, model in enumerate(models):
        model_id = model.modelId
        print(f"\n[{idx+1}/{LIMIT}] 正在处理：{model_id}")
        result = {
            # 基础元数据
            "model_id": model_id,
            "author": model.author if model.author else "未知",
            "downloads": model.downloads if model.downloads else 0,
            "likes": model.likes if model.likes else 0,
            "last_modified": str(model.lastModified) if model.lastModified else "未知",
            "library_name": model.library_name if model.library_name else "未知",
            "pipeline_tag": model.pipeline_tag if model.pipeline_tag else "未知",
            "tags": ",".join(model.tags) if model.tags else "无",

            # 12层提取来源溯源
            "l1_card": "无",
            "l2_main_config": "无",
            "l3_separate_config": "无",
            "l4_diffusers_config": "无",
            "l5_readme": "无",
            "l6_tags": "无",
            "l7_model_id": "无",
            "l8_model_type": "无",
            "l9_safetensors_index": "无",

            # 供应链核心字段
            "base_model_final": "基座模型",
            "is_base_model_valid": False,
            "is_multimodal": False,
            "dataset_deps": [],
            "extract_source": "无",
            "status": "success",
            "error_msg": ""
        }

        try:
            # 获取模型完整信息
            model_info = api.model_info(model_id, timeout=TIMEOUT)
            card_data = model_info.cardData or {}
            model_config = model_info.config or {}
            model_type = model_config.get("model_type", "")

            # --------------------------
            # 逐层执行提取
            # --------------------------
            result["l1_card"] = extract_base_from_card(card_data)
            result["l2_main_config"], result["is_multimodal"] = extract_base_from_main_config(model_id)
            result["l3_separate_config"] = extract_base_from_separate_config(model_id)
            result["l4_diffusers_config"] = extract_base_from_diffusers_config(model_id)
            result["l5_readme"] = extract_base_from_readme(model_id)
            result["l6_tags"] = extract_base_from_model_tags(model.tags)
            result["l7_model_id"] = extract_base_from_model_id(model_id)
            result["l8_model_type"] = extract_base_from_model_type(model_type)
            result["l9_safetensors_index"] = extract_base_from_safetensors_index(model_id)

            # 数据集依赖
            result["dataset_deps"] = card_data.get("datasets", [])

            # --------------------------
            # 最终优先级决策（从高到低，越靠前越可靠）
            # --------------------------
            extract_priority = [
                ("l1_card", "模型卡片显式标注"),
                ("l2_main_config", "主配置文件"),
                ("l3_separate_config", "独立Adapter配置"),
                ("l9_safetensors_index", "Safetensors索引"),
                ("l5_readme", "README文档"),
                ("l6_tags", "模型标签"),
                ("l4_diffusers_config", "Diffusers组件配置"),
                ("l7_model_id", "模型ID语义解析"),
                ("l8_model_type", "模型类型映射"),
            ]

            # 选出最终有效父模型
            final_base = None
            final_source = None
            for key, source_name in extract_priority:
                candidate = result[key]
                if candidate and candidate not in ["无", ""]:
                    final_base = candidate
                    final_source = source_name
                    break

            # 多模态兜底
            if final_base is None and result["is_multimodal"]:
                final_base = "多模态内部组件"
                final_source = "多模态配置识别"

            # 最终赋值
            result["base_model_final"] = final_base if final_base else "基座模型"
            result["extract_source"] = final_source if final_source else "官方基座模型"

            # --------------------------
            # 父模型有效性校验
            # --------------------------
            if VERIFY_BASE_MODEL:
                result["is_base_model_valid"] = verify_model_exists(result["base_model_final"])

            success_count += 1
            print(f"  ✅ 提取完成 | 父模型：{result['base_model_final']} | 来源：{result['extract_source']}")

        except Exception as e:
            failed_count += 1
            result["status"] = "failed"
            result["error_msg"] = str(e)[:100]
            print(f"  ❌ 处理失败 | 原因：{result['error_msg']}")

        model_data.append(result)
        time.sleep(SLEEP_TIME)

    # 4. 结果保存
    df_models = pd.DataFrame(model_data)
    # 主文件
    df_models.to_csv(f"{OUTPUT_DIR}/hf_models_ultimate.csv", index=False, encoding="utf-8-sig")
    df_models.to_json(f"{OUTPUT_DIR}/hf_models_ultimate.json", orient="records", force_ascii=False, indent=4)
    # 成功提取的有效数据子集
    df_valid = df_models[df_models["status"] == "success"]
    df_valid.to_csv(f"{OUTPUT_DIR}/hf_models_valid.csv", index=False, encoding="utf-8-sig")

    # 5. 最终统计
    print(f"\n" + "="*50)
    print(f"🏁 终极版抓取完成！")
    print(f"📊 总模型数：{LIMIT} | 成功：{success_count} | 失败：{failed_count}")
    print(f"🎯 有效父模型提取率：{round(len(df_valid[df_valid['base_model_final']!='基座模型'])/len(df_valid)*100, 2)}%")
    print(f"📁 结果已保存到 {OUTPUT_DIR}/ 目录")
    print("="*50)