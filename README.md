# LLM-Genealogy

展示 Hugging Face 模型之间的谱系关系，并支持在首页手动触发更新。

## 新增能力

- `index.html` 可以直接输入抓取数量并点击按钮更新
- 手动输入的抓取数量会持久化，后续每周更新沿用这个数量
- 后续每周更新采用增量模式，只新增和移除与上次结果不同的模型
- 保留原有 `graph.html` 和 `table.html` 页面

## 本地启动

```bash
pip install -r requirements.txt
python server.py
```

然后访问：

```text
http://127.0.0.1:8000
```

## 只执行一次更新

```bash
python server.py --once
```

## 自动更新

- 本地服务运行期间，会根据保存的抓取数量按周检查是否需要更新
- GitHub Actions 也已改为每周执行一次同样的增量更新逻辑
