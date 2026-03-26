import pandas as pd

# 读取有效模型数据
df = pd.read_csv("data/hf_models_valid.csv")

# ======================
# 步骤1：过滤出【有基座模型】的模型
# ======================
df_has_base = df[
    (df["base_model_final"] != "基座模型") &
    (df["base_model_final"] != "获取失败") &
    (df["base_model_final"] != "多模态内部组件")
].copy()

print(f"过滤前总数：{len(df)}")
print(f"有基座模型的模型数量：{len(df_has_base)}")

# ======================
# 步骤2：提取所有依赖的基座模型 ID
# ======================
base_model_ids = df_has_base["base_model_final"].dropna().unique().tolist()

# 从原始数据中匹配已存在的基座模型
df_exist_bases = df[df["model_id"].isin(base_model_ids)].copy()

# 找出原始数据中不存在的基座模型
missing_bases = [bid for bid in base_model_ids if bid not in df["model_id"].values]

# ======================
# 步骤3：补全缺失的基座模型（未知字段填 未知）
# ======================
missing_rows = []
for model_id in missing_bases:
    row = {
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
        "base_model_final": "基座模型",
        "is_base_model_valid": "未知",
        "is_multimodal": "未知",
        "dataset_deps": "未知",
        "extract_source": "未知",
        "status": "success",
        "error_msg": ""
    }
    missing_rows.append(row)

df_missing_bases = pd.DataFrame(missing_rows)

# ======================
# 步骤4：合并所有数据并去重
# ======================
df_final = pd.concat(
    [df_has_base, df_exist_bases, df_missing_bases],
    ignore_index=True
)

# 按 model_id 去重，保留第一条
df_final = df_final.drop_duplicates(subset=["model_id"], keep="first")

# ======================
# 步骤5：保存（已修复编码错误）
# ======================
df_final.to_csv("hf_models_has_base.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ 最终总数量（模型+基座）：{len(df_final)}")
print("✅ 已保存到：hf_models_has_base.csv")
print("\n📌 可直接用于 Neo4j 图谱构建 / 模型血缘分析")



