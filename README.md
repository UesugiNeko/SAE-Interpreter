# SAE Interpreter

一个面向 SAE（Sparse Autoencoder）特征分析与解释的本地项目，提供：

- 基于 `.pt/.json` 的已有结果加载与可视化
- 基于模型 + SAE + 数据集的在线推断（Infer）
- 多种特征解释方法
- Token 搜索相似特征、Prompt Activation、Feature Steering
- Streamlit 前端交互页面

本项目在工程结构上参考了 OpenAI `neuron-explainer`，并结合 `sae_dashboard/sae_lens` 进行了落地集成。

---

## 1. 环境准备

推荐使用已配置的 `saebench` 环境。

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 启动前端

```bash
streamlit run sae_interpretability_app.py --server.headless true --server.address 127.0.0.1 --server.port 8510
```

---

## 2. 使用流程

### 2.1 Load existing data

适用于你已经有：

- `*_activations.pt` / `*.json`
- `*_logits.pt`（可选但推荐）

加载后可直接进行：

- Feature Explorer（logits + sequence + prompt activation）
- Explain（多解释方法）
- Token Search
- Steer

### 2.2 Infer now

输入：

- `model path`
- `tokenizer path`
- `sae path`
- `dataset path`

在线推断后会构建：

- `feat_to_logits`
- `token_to_feat`
- feature sequence activations

并进入同样的分析页面。

---

## 3. 解释方法说明

- `np_max-act-logits`（Top Logits+激活模式）
- `token-activation-pair`（上下文+激活模式）
- `token-space-representation`（token集合的语义共性）

解释模式支持：

- `heuristic`（纯规则）
- `api`（调用 LLM API）
- `auto`（有 key 走 API，否则 heuristic）

---

## 4. SAE Dashboard 相关用法（重点）

项目在 Infer 流程中使用了 `sae_dashboard` 的核心组件：

- `SaeVisConfig`
- `SaeVisRunner`
- `ActivationsStore.from_sae`
- `get_tokens`

---

## 5. 输入/输出数据格式

### 5.1 activations（按特征组织）

每个 feature 对应一个序列列表，单条记录至少包含：

- `tokens: List[str]`
- `feat_acts: List[float]`

可选：

- `token_ids: List[int]`

### 5.2 logits

推荐包含：

- `feat_to_logits`: `feature -> {positive, negative}`
- `token_to_feat`: `{"indices": Tensor, "values": Tensor}`

---

## 6. 设备与显存说明

- 前端提供 `Device` 选择（`auto/cpu/cuda:i`）
- `auto` 会按后端策略挑选可用卡
- 任务结束后会调用显存释放流程（`empty_cache/ipc_collect`）

提示：CUDA context 的少量常驻显存属于正常现象。

---

## 7. 致谢

- OpenAI `neuron-explainer`
- `sae_lens`
- `sae_dashboard`

