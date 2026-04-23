# SAE Interpreter

一个面向 SAE（Sparse Autoencoder）特征分析与解释的本地项目，提供：

- 基于 `.pt/.json` 的已有结果加载与可视化
- 基于模型 + SAE + 数据集的在线推断（Infer）
- 多种特征解释方法（含 `np_max-act-logits`、`token-activation-pair`、`token-space-representation`）
- Token 搜索相似特征、Prompt Activation、Feature Steering
- Streamlit 前端交互页面

本项目在工程结构上参考了 OpenAI `neuron-explainer`，并结合 `sae_dashboard/sae_lens` 进行了落地集成。

---

## 1. 主要文件

- `sae_interpretability_app.py`：Streamlit 前端入口
- `sae_ui_backend.py`：后端核心逻辑（加载、推断、解释、steer、token 搜索）
- `np_max_act_logits_interpreter.py`：解释方法与批量解释脚本
- `logitlens_test.py`：离线构建缓存/激活数据的实验与导出脚本
- `requirements.txt`：依赖列表

---

## 2. 环境准备

推荐使用已配置的 `saebench` 环境。

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 启动前端

```bash
streamlit run sae_interpretability_app.py --server.headless true --server.address 127.0.0.1 --server.port 8510
```

---

## 3. 使用流程

### 3.1 Load existing data

适用于你已经有：

- `*_activations.pt` / `*.json`
- `*_logits.pt`（可选但推荐）

加载后可直接进行：

- Feature Explorer（logits + sequence + prompt activation）
- Explain（多解释方法）
- Token Search
- Steer

### 3.2 Infer now

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

## 4. 解释方法说明

当前集成的方法位于 `np_max_act_logits_interpreter.py`：

- `np_max-act-logits`
- `token-activation-pair`
- `token-space-representation`

解释模式支持：

- `heuristic`（纯规则）
- `api`（调用 LLM API）
- `auto`（有 key 走 API，否则 heuristic）

---

## 5. SAE Dashboard 相关用法（重点）

项目在 Infer 流程中直接使用了 `sae_dashboard` 的核心组件：

- `SaeVisConfig`
- `SaeVisRunner`
- `ActivationsStore.from_sae`
- `get_tokens`

对应代码入口：`sae_ui_backend.py -> infer_feature_data_from_model(...)`

Prompt Activation 也已对齐 SAE Dashboard 的关键习惯：

- 使用 `prepend_bos=False`
- 优先走 `FeatureDataGeneratorFactory` 路径提取激活
- 失败时回退到 `run_with_cache + sae.encode`

---

## 6. 输入/输出数据格式

### 6.1 activations（按特征组织）

每个 feature 对应一个序列列表，单条记录至少包含：

- `tokens: List[str]`
- `feat_acts: List[float]`

可选：

- `token_ids: List[int]`

### 6.2 logits

推荐包含：

- `feat_to_logits`: `feature -> {positive, negative}`
- `token_to_feat`: `{"indices": Tensor, "values": Tensor}`

---

## 7. 设备与显存说明

- 前端提供 `Device` 选择（`auto/cpu/cuda:i`）
- `auto` 会按后端策略挑选可用卡
- 任务结束后会调用显存释放流程（`empty_cache/ipc_collect`）

提示：CUDA context 的少量常驻显存属于正常现象。

---

## 8. 注意事项

- 建议将大权重文件加入 `.gitignore`（如 `*.pt`, `*.ptwe`, `*.safetensors`）
- 前端触发的长任务建议在稳定网络环境下执行
- 若需“断开网页仍持续跑”，建议改用 tmux + 脚本命令行方式执行批处理

---

## 9. 致谢

- OpenAI `neuron-explainer`
- `sae_lens`
- `sae_dashboard`

