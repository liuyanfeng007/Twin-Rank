# Twin-Rank 使用说明

> 设计思路与取舍见 [ARCHITECTURE.md](ARCHITECTURE.md)；项目亮点与演示脚本见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)。

双塔向量召回 + LightGBM 精排 + MMR 重排。本文说明：怎么跑、每个参数什么意思、怎么换成自己的数据、怎么注册模型并调用预测。

目录：
1. [整体流程](#1-整体流程) 2. [平台与环境](#2-平台与环境) 3. [快速开始](#3-快速开始) 4. [配置参数详解](#4-配置参数详解)
5. [使用自己的数据](#5-使用自己的数据) 6. [MLflow：记录与模型注册](#6-mlflow记录与模型注册) 7. [调用模型预测](#7-调用模型预测)
8. [输出字段说明](#8-输出字段说明) 9. [常见问题](#9-常见问题) 10. [代码结构](#10-代码结构)

---

## 1. 整体流程

```
 原始数据 ──► ① 下载/读取 ──► ② 清洗+按时间切分 ──► ③ 特征工程 ──► ④ 双塔训练(MLflow)
                                                                         │ 物品向量
 Reflex Demo ◄── ⑧ FastAPI ◄── ⑦ MMR 重排 ◄── ⑥ LightGBM 精排 ◄── ⑤ Faiss 索引 + 多路召回
                    ▲
                    └── ⑩ 注册到 MLflow Model Registry（整套打包）──► 任意机器 load_model().predict()
```

| 步骤 | 命令 | 输入 → 输出 |
|---|---|---|
| ① 下载 | `make download` | 网络 → `data/raw/*.jsonl.gz`（自定义数据时跳过） |
| ② 预处理 | `make preprocess` | raw → `data/processed/interactions.parquet`、`items_meta.parquet` |
| ③ 特征 | `make features` | processed → `data/features/*`（样本、编码器、物品特征） |
| ④ 双塔 | `make train` | features → `data/artifacts/two_tower.pt`、`item_embeddings.npy` |
| ⑤ 索引 | `make index` | 物品向量 → `data/artifacts/items.faiss` |
| ⑥ 精排 | `make ranker` | 召回候选 → 精排模型文件（按 `ranker.model_type`），并输出端到端指标 |
| ⑩ 注册 | `make register` | 整套产物 → MLflow Model Registry |
| ⑧ 服务 | `make serve` | FastAPI（默认 8100 端口，文档在 `/docs`） |
| ⑨ 演示 | `make app` | Reflex 页面（默认 3000 端口） |

> `make pipeline` 依次执行 ①–⑥ 和 ⑩。

## 2. 平台与环境

| 系统 | 支持情况 | 说明 |
|---|---|---|
| macOS（Intel / Apple Silicon） | ✔ | 需要 `brew install libomp`（启动脚本会自动处理）。默认单线程避免 libomp 冲突；Apple GPU(MPS) 会被自动用于训练 |
| Linux（x86_64 / aarch64） | ✔ | Debian/Ubuntu 需 `python3-venv`、`libgomp1`、`unzip`（Reflex 用）。有 NVIDIA GPU 时自动用 CUDA |
| Windows 10/11 | ✔ | 使用 `tools\start.bat` |

* Python：推荐 **3.9–3.12**（太新的版本可能没有 torch/faiss 的预编译包）。
* 运行 `python tools/doctor.py` 可以自检环境，它会针对你的系统给出具体修复命令。
* MLflow：需要一个已启动的 MLflow 服务（`mlflow server --host 127.0.0.1 --port 5000`），并设置 `MLFLOW_TRACKING_URI`。不设置时使用本地 `./mlruns`。**模型注册需要数据库后端**（如 sqlite：`--backend-store-uri sqlite:///mlflow.db`），纯文件后端在新版本 MLflow 里不支持注册表。

## 3. 快速开始

**一键（推荐）**
```bash
tools/start.sh            # macOS / Linux；强制重训加 --retrain
tools\start.bat           # Windows；强制重训加 /retrain
```
脚本流程：建虚拟环境 → 装依赖 → 环境自检 → 跑完整流水线（含注册模型）→ 启动 API → 启动 Demo。首次运行较慢（下载数据+训练），之后会跳过。

**分步**
```bash
pip install -e ".[demo,dev]"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make download preprocess features train index ranker register
make serve        # 终端 1
make app          # 终端 2
```

**不联网/快速体验（合成数据，几分钟跑完）**
```bash
python tools/make_demo_custom_data.py          # 生成 data/custom/{interactions,items}.csv
C=configs/custom_demo.yaml
for s in preprocess_data build_features train_two_tower build_index train_ranker register_model; do
  python -m twin_rank.pipelines.$s --config $C; done
```

## 4. 配置参数详解

所有参数在 `configs/config.yaml`（每一项都带注释）。任何脚本都支持 `--config 路径` 使用其他配置文件。

### 4.1 `data` 数据来源与目录
| 参数 | 默认 | 说明 |
|---|---|---|
| `source` | `amazon` | `amazon` 下载公开数据；`custom` 使用你自己的数据（见第 5 节） |
| `category` | `Video_Games` | Amazon 类目。想快速试跑选小类目，如 `Digital_Music`、`Musical_Instruments` |
| `review_url` / `meta_url` | 官方地址 | 下载地址模板，官方路径变化时在这里改 |
| `raw_dir` `processed_dir` `feature_dir` `artifact_dir` | `data/...` | 各阶段产物目录。**换数据集时请改成不同目录**，避免新旧产物混在一起 |
| `custom.*` | — | 自定义数据的文件路径与列名映射，见第 5 节 |

### 4.2 `preprocess` 预处理
| 参数 | 默认 | 说明 |
|---|---|---|
| `positive_rating` | 4.0 | 评分 ≥ 该值才算"喜欢"（正样本）。自定义数据没有评分列时全部算正样本 |
| `min_user_interactions` | 5 | 交互数少于该值的用户被剔除。至少要 3 才能切出训练/验证/测试 |
| `min_item_interactions` | 5 | 交互数少于该值的商品被剔除。调小会保留更多长尾商品，但这些商品学不好 |
| `max_review_rows` | null | 只读前 N 条评论（调试用，如 200000） |

### 4.3 `features` 特征
| 参数 | 默认 | 说明 |
|---|---|---|
| `max_history_len` | 20 | 用户历史保留最近多少个商品。越长越能反映长期兴趣，也越占内存 |
| `top_stores` | 2000 | 只保留最常见的 N 个品牌，其余归为 other |
| `n_price_buckets` / `n_pop_buckets` | 10 | 价格 / 热度的分位数分桶数 |

### 4.4 `model` 双塔结构
| 参数 | 默认 | 说明 |
|---|---|---|
| `embedding_dim` | 64 | ID embedding 维度。数据少时调小（32）防过拟合；数据多可调大（128） |
| `side_dim` | 16 | 类目/品牌/价格/热度 embedding 维度 |
| `hidden_dims` | [128] | 塔内 MLP 隐藏层；`[256,128]` 表示两层 |
| `output_dim` | 64 | 用户/物品向量维度，同时是 Faiss 索引维度 |
| `temperature` | 0.05 | softmax 温度。越小对难负样本越敏感，但训练更不稳定；常用 0.05–0.2 |

### 4.5 `train` 训练
| 参数 | 默认 | 说明 |
|---|---|---|
| `batch_size` | 1024 | 同时也是 batch 内负样本数量。越大负样本越多、效果通常越好，但更慢更占内存 |
| `epochs` | 20 | 最大轮数（有早停） |
| `lr` / `weight_decay` | 1e-3 / 1e-6 | Adam 学习率 / L2 正则 |
| `eval_ks` | [10,50,100] | 评估的 K 值 |
| `monitor_metric` | `recall_at_50` | 早停与选最佳模型依据；必须是 `recall_at_K` 或 `ndcg_at_K` 且 K 在 `eval_ks` 里 |
| `patience` | 3 | 连续几轮验证无提升就停止 |
| `device` | `auto` | `auto` 依次尝试 CUDA → Apple MPS → CPU；也可强制 `cpu`/`cuda`/`mps` |
| `seed` | 42 | 随机种子 |

### 4.6 `retrieval` 召回
| 参数 | 默认 | 说明 |
|---|---|---|
| `index_type` | `flat` | `flat` 精确检索（商品 < 100 万够用）；`hnsw` 近似检索，更快但略损精度 |
| `hnsw_m` / `hnsw_ef_search` | 32 / 128 | 仅 hnsw：图连接数 / 搜索宽度，越大越准越慢 |
| `n_candidates` | 100 | 召回候选总数。**这是精排效果的上限**（候选里没有的商品精排救不回来），调大会提高上限但增加精排耗时 |
| `n_popular` | 10 | 候选里由"热门路"补充的条数，其余来自双塔 |

### 4.7 `ranker` 精排模型（LightGBM / DeepFM / DCN / DIN）
| 参数 | 默认 | 说明 |
|---|---|---|
| `model_type` | `lightgbm` | 精排模型：`lightgbm` / `deepfm` / `dcn` / `din`。怎么选见 [4.7.1](#471-精排模型怎么选)。可用环境变量 `TWIN_RANK_RANKER` 临时覆盖 |
| `objective` | `lambdarank` | `lambdarank` 直接优化排序；`binary` 当点击率二分类 |
| `max_train_users` | 30000 | 用多少个验证集用户构造精排样本。调试可设 5000，正式训练尽量大 |
| `valid_ratio` | 0.2 | 其中用于精排验证的比例 |
| `eval_users` | 10000 | 用多少测试用户做端到端评估 |
| `eval_ks` | [10,20,50] | 端到端评估的 K |
| `num_boost_round` / `early_stopping_rounds` | 500 / 30 | 最大树数 / 早停轮数 |
| `model_name` | `lightgbm_ranker.txt` | 精排模型文件名 |
| `params.*` | — | 仅 lightgbm：原样传给 LightGBM（`learning_rate`、`num_leaves`、`min_data_in_leaf`、`feature_fraction`、`bagging_fraction`、`bagging_freq`） |
| `deep.embedding_dim` | 16 | 仅深度模型：每个特征字段的 embedding 维度 |
| `deep.hidden_dims` | [128, 64] | 仅深度模型：DNN 隐藏层 |
| `deep.dropout` | 0.1 | 防过拟合；训练样本少时可调大到 0.3 |
| `deep.cross_layers` | 3 | 仅 dcn：交叉层数，最高交叉阶数 = 层数 + 1 |
| `deep.attention_hidden` | [64, 32] | 仅 din：注意力网络隐藏层 |
| `deep.batch_size` / `epochs` / `lr` / `weight_decay` | 2048 / 10 / 1e-3 / 1e-6 | 训练超参 |
| `deep.patience` | 2 | 验证 NDCG@10 连续几轮无提升则早停 |
| `deep.device` | `auto` | `auto` / `cpu` / `cuda` / `mps`；几十万行样本用 CPU 就够 |
| `deep.model_name` | `deep_ranker_{model_type}.pt` | 模型文件名，`{model_type}` 自动替换，三种深度模型各存各的 |

> **注意**：只有"候选里包含真实商品"的用户才参与精排训练（约 20%）。`candidate_hit_rate` 偏低时，先提升召回比调精排参数更有效。

### 4.7.1 精排模型怎么选

四种模型**输入完全相同**（同一份召回候选、同一套精排特征），输出也相同（每个候选一个分数），所以可以只改一行配置互相替换，并且在同一份数据上公平对比。

| | LightGBM | DeepFM | DCN（V2） | DIN |
|---|---|---|---|---|
| 原理 | 梯度提升树 | FM 二阶交叉 + DNN | 显式多阶交叉网络 + DNN | 候选商品对用户历史做注意力 + DNN |
| 特征交叉 | 树分裂隐式学习 | 二阶显式 + 高阶隐式 | 高阶显式（阶数 = `cross_layers`+1）+ 隐式 | 同 DNN，另外会按候选挑选相关历史 |
| 用到用户历史序列 | 否（只用统计特征） | 否 | 否 | **是**（最近 `max_history_len` 个商品） |
| 商品 ID 的 embedding | 否 | 是 | 是 | 是 |
| 所需训练数据量 | 少即可 | 较多 | 较多 | 最多 |
| 训练速度 | 快（秒～分钟） | 较慢 | 较慢 | 最慢（注意力开销） |
| 硬件 | CPU | CPU 可，GPU 更快 | 同左 | 同左 |
| 推荐理由 | 特征贡献（精确） | 遮挡法（近似） | 遮挡法（近似） | 遮挡法（近似） |
| 过拟合风险 | 低 | 中 | 中 | 较高 |
| 损失函数 | LambdaRank（直接优化排序）或二分类 | 逐点二分类（BCE） | 逐点二分类（BCE） | 逐点二分类（BCE） |

**选择建议**（以下是行业常见经验，不是在你的数据上测出来的结论，请以第 5 项的对比实验为准）：

| 你的情况 | 建议 | 原因 |
|---|---|---|
| 第一次跑通 / 数据不多（日志里参与精排训练的用户不到约 1 万） | **LightGBM** | 小数据上最稳，训练快，不依赖 GPU，不易过拟合 |
| 特征主要是统计和匹配类，要求推荐理由准确、线上延迟低 | **LightGBM** | 树模型对这类数值特征很强，特征贡献可直接解释 |
| 数据量中等以上，想让模型自动学习"类目 × 品牌 × 价格档位"这类交叉 | **DeepFM** | 同时有低阶和高阶交叉，结构简单、训练稳定 |
| 数据量较大，特征交叉重要，DeepFM 提升已到瓶颈 | **DCN** | 显式高阶交叉，参数利用更充分 |
| 用户历史较长且兴趣分散（同一人买很多不同类的商品），数据量大 | **DIN** | 针对每个候选挑选相关的历史，平均池化做不到这一点 |
| 不确定 | 先用 **LightGBM** 当基线，再逐个试深度模型，用同一份数据对比 | 深度模型只有在数据量足够时才可能超过 LightGBM |

**一个重要提醒**：深度模型在样本少时往往不如 LightGBM。精排只用"候选里含目标"的用户训练，比例通常只有约 20%，所以实际样本比 `max_train_users` 小很多。想让深度模型发挥，先把 `max_train_users` 调大。

**怎么切换**
```bash
# 方式 1：改 configs/config.yaml
ranker:
  model_type: din

# 方式 2：环境变量临时覆盖，不改文件（适合对比实验）
TWIN_RANK_RANKER=din python -m twin_rank.pipelines.train_ranker
```
切换后需要依次执行：`train_ranker`（训练）→ `register_model`（注册，新版本里就是新的精排）。在线服务和预测读取的是 `model_type` 对应的文件，所以 `model_type` 要与已训练的类型一致，否则会提示找不到模型文件并告诉你怎么处理。

**一次对比四个模型**
```bash
make compare-rankers                       # macOS / Linux
# 或手动：
for t in lightgbm deepfm dcn din; do TWIN_RANK_RANKER=$t python -m twin_rank.pipelines.train_ranker; done
# Windows (cmd):
for %t in (lightgbm deepfm dcn din) do (set TWIN_RANK_RANKER=%t && python -m twin_rank.pipelines.train_ranker)
```
在 MLflow 里，每个模型是一个 run（名字形如 `ranker-v1-din`，并带有标签 `ranker_type`）。对比这几项：

| 指标 | 含义 | 怎么看 |
|---|---|---|
| `test_ranker_ndcg_at_10`、`test_ranker_recall_at_10` | 召回 + 精排的端到端效果 | **主要依据**，越大越好 |
| `test_full_ndcg_at_10` | 再加上重排之后 | 最终线上效果的近似 |
| `test_recall_only_*` | 仅召回 | 四个 run 应当几乎一样，用来确认比较是公平的 |
| `valid_ndcg_at_10` 曲线 | 精排自己的验证曲线 | 深度模型看是否过早下降（过拟合） |
| `test_candidate_hit_rate` | 候选命中率 | 精排效果上限，与精排模型无关 |

判断时注意：`eval_users` 太小时指标会有波动，建议 ≥ 10000；差异小于约 0.002 可视为持平，这时选更简单、更快的模型。

**其他说明**
* 不同类型的模型文件分开保存（`lightgbm_ranker.txt`、`deep_ranker_deepfm.pt`、`deep_ranker_dcn.pt`、`deep_ranker_din.pt`），所以对比时不会互相覆盖。
* 深度模型输出的 `rank_score` 是点击概率（0–1），LightGBM 的 lambdarank 分数没有固定范围，两者的数值不能互相比较，只比较排序效果。
* 深度模型的推荐理由用"遮挡法"近似，只覆盖数值特征，不如 LightGBM 的特征贡献精确。

### 4.8 `rerank` 重排与业务规则
| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | 关闭后直接按精排分取 Top-K |
| `mmr_lambda` | 0.7 | 相关性 vs 多样性：1 = 只看相关性，0 = 只看多样性 |
| `max_same_store` | 3 | 同一品牌最多出现几次，0 = 不限制 |
| `blacklist_items` | [] | 黑名单（原始商品 ID），永不推荐 |
| `pinned_items` | [] | 运营置顶（原始商品 ID），固定排最前；用户看过的自动跳过 |

### 4.9 `serving` 在线服务
| 参数 | 默认 | 说明 |
|---|---|---|
| `host` / `port` | 0.0.0.0 / 8100 | API 监听地址 |
| `sample_users` | 50 | Demo 下拉框展示的示例用户数 |
| `model_uri` | null | 空 = 读本地 `data/`；填 `models:/twin-rank-recommender@champion` = 从 MLflow 注册表加载。也可用环境变量 `TWIN_RANK_MODEL_URI` |
| `cache.enabled` / `ttl_seconds` | true / 300 | 推荐结果缓存开关与过期时间 |
| `cache.redis_url` | null | 如 `redis://localhost:6379/0`；为空或连接失败则用进程内缓存 |

### 4.10 `mlflow` 与 `registry`
| 参数 | 默认 | 说明 |
|---|---|---|
| `mlflow.tracking_uri` | null | 空则读环境变量 `MLFLOW_TRACKING_URI` |
| `mlflow.experiment_name` | `twin-rank` | 实验名 |
| `mlflow.run_name` / `ranker_run_name` | — | 双塔 / 精排训练 run 的名字 |
| `registry.model_name` | `twin-rank-recommender` | 注册表中的模型名 |
| `registry.alias` | `champion` | 线上使用的别名；引用方式 `models:/<名字>@<别名>` |
| `registry.challenger_alias` / `previous_alias` | `challenger` / `previous` | 未晋升的新版本别名 / 上一任 champion 别名（回滚用） |
| `registry.promote_metric` | `test_full_ndcg_at_10` | 决定是否晋升的端到端指标，须是 `train_ranker` 记录过的 `test_*` 指标 |
| `registry.min_improvement` | 0.0 | 新版本至少要高出 champion 多少才晋升，防止被随机波动误导 |

### 4.11 环境变量
| 变量 | 作用 |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow 服务地址 |
| `TWIN_RANK_MODEL_URI` | 让 API 从注册表加载模型 |
| `TWIN_RANK_RANKER` | 临时覆盖 `ranker.model_type`（`lightgbm` / `deepfm` / `dcn` / `din`），所有入口脚本都生效 |
| `TWIN_RANK_API_URL` | Reflex 前端访问的 API 地址（默认 `http://127.0.0.1:8100`） |
| `OMP_NUM_THREADS` | macOS 默认 1；想提速可设 4，卡住就改回 1 |
| `SSL_CERT_FILE` / `TWIN_RANK_INSECURE_SSL=1` | 下载时的证书问题（公司代理场景） |
| `API_PORT` `PYTHON` `SKIP_PIPELINE` | 启动脚本用 |

## 5. 使用自己的数据

### 5.1 准备数据
只需要一张**行为表**（必需），可选再给一张**商品表**。支持 `csv / tsv / parquet / jsonl`（csv 可 gz 压缩）。

行为表（每行 = 一次"用户喜欢某商品"的行为）：
| 列 | 必需 | 说明 |
|---|---|---|
| 用户 ID | ✔ | 任意字符串/数字，程序会统一转成字符串 |
| 商品 ID | ✔ | 同上 |
| 时间 | ✔ | 数字（秒/毫秒时间戳）或日期字符串（`2024-05-01 12:00:00`）；只用于排序 |
| 评分 | 可选 | 没有就写 `null`，全部行为视为正样本；有则 ≥ `positive_rating` 才算正样本 |

商品表（可选，提供越多特征越好）：商品 ID（必需）、标题、类目、品牌、价格。缺哪列就留空，对应特征退化为"未知"。

> 想看示例：`python tools/make_demo_custom_data.py` 会在 `data/custom/` 生成一份格式正确的样例文件。

### 5.2 修改配置
建议复制一份配置：`cp configs/config.yaml configs/my_data.yaml`，然后修改：
```yaml
data:
  source: custom
  processed_dir: data/my/processed      # 换成独立目录，别和 Amazon 的产物混用
  feature_dir:   data/my/features
  artifact_dir:  data/my/artifacts
  custom:
    interactions_file: /path/to/我的行为.csv
    items_file: /path/to/我的商品.csv    # 没有就写 null
    columns:                             # 右边写"你的文件里的列名"
      user_id: uid
      item_id: sku
      timestamp: created_at
      rating: null
    item_columns:
      item_id: sku
      title: name
      category: cate
      store: brand
      price: price
registry:
  model_name: my-recommender             # 与 Amazon 版区分开
mlflow:
  experiment_name: my-recommender
```

### 5.3 运行
```bash
C=configs/my_data.yaml
for s in preprocess_data build_features train_two_tower build_index train_ranker register_model; do
  python -m twin_rank.pipelines.$s --config $C; done
TWIN_RANK_CONFIG=$C python -m twin_rank.pipelines.serve_api --config $C    # 可选：启动 API
```
自定义数据**不需要** `download_data`。

### 5.4 数据量与调参建议
| 情况 | 建议 |
|---|---|
| 用户 < 1 万 / 交互 < 50 万 | `embedding_dim: 32`、`batch_size: 256`、`epochs` 可适当加大；`min_*_interactions` 调到 3 |
| 商品数很少（< 1000） | `n_candidates` 调小（如 50），`index_type: flat` |
| 商品数很多（> 100 万） | `index_type: hnsw` |
| 日志里 `candidate_hit_rate` 很低 | 先改善召回：增大 `n_candidates`、`epochs`、`embedding_dim` |
| 没有商品表 | 只靠 ID 与行为序列，效果略低，仍可正常运行 |

### 5.5 常见报错
* `行为表缺少列 [...]`：`data.custom.columns` 里写的列名与文件不一致，报错信息会列出文件的实际列名。
* 清洗后交互为 0：`positive_rating` 太高，或 `min_*_interactions` 太大，调小即可。
* ID 是数字（如 `12345`）也没问题，程序会统一当作字符串；预测时 `user_id` 也传字符串。

## 6. MLflow：记录与模型注册

### 6.1 训练记录
`train_two_tower` 与 `train_ranker` 会各自产生一个 run，记录参数、逐轮指标（`val_recall_at_50` 等）、测试指标（`test_*`）、模型文件。端到端指标命名规则：`test_{recall_only|ranker|full}_{recall|ndcg}_at_{K}`，`test_candidate_hit_rate` 是候选命中率。

### 6.2 注册整套模型（带质量门禁）
```bash
python -m twin_rank.pipelines.register_model                  # 注册新版本，指标更好才晋升 champion
python -m twin_rank.pipelines.register_model --force          # 无视比较，强制晋升
python -m twin_rank.pipelines.register_model --alias staging  # 直接指定别名，不做比较
```
注册的是**整套推荐系统**（用户塔 + 物品向量 + Faiss 索引 + 精排模型 + ID 编码器 + 物品特征 + 商品目录 + 用户历史快照 + `twin_rank` 源码），而不只是某一个模型文件。这样 `v3` 加载出来就是彼此一致、可直接预测的完整系统。

**最新 ≠ 最好**：每次注册都会产生新版本（保留完整历史），但是否替换线上的 `champion`，由指标决定：

| 情况 | 结果 |
|---|---|
| 还没有 champion | 直接晋升 |
| 新版本的 `registry.promote_metric` 超过 champion（并且高出 `min_improvement`） | 晋升：champion 指向新版本，原 champion 留作 `previous` |
| 没超过 champion | **不晋升**，新版本只打 `challenger` 别名，线上继续用旧的 champion |
| 与 champion 评估口径不同（数据集、测试用户数、K 值、随机种子任一不同） | 不能比较，**不自动晋升**，打 `challenger`，由人判断 |
| champion 是旧流程注册的（没有指标标签） | 不能比较，同上 |

指标来源：`train_ranker` 结束时把端到端测试指标、评估口径和模型文件指纹写入 `ranker_metrics_<类型>.json`，注册时读取并写成版本标签（`metric.*`、`fingerprint`、`ranker_type`）。注册前会校验指纹：如果评估之后双塔或精排又被重训过，会报错并要求重新运行 `train_ranker`，保证"指标"和"被注册的模型"是同一份。

门禁相关配置（`registry.*`）：`promote_metric`（默认 `test_full_ndcg_at_10`，是加上重排之后的最终效果）、`min_improvement`（至少高出多少才晋升，指标有随机波动，建议 0.001 左右）。

### 6.3 版本管理
```bash
python -m twin_rank.pipelines.promote_model --list        # 列出各版本：别名、精排类型、对比指标、注册时间
python -m twin_rank.pipelines.promote_model --version 3   # 手动把版本 3 设为 champion
python -m twin_rank.pipelines.promote_model --rollback    # 回滚：champion 回到 previous
```
| 别名 | 含义 |
|---|---|
| `champion` | 线上在用的版本（预测、API、Demo 默认用它） |
| `challenger` | 新训练、但没超过 champion 的版本；可用 `models:/<名字>@challenger` 单独试用 |
| `previous` | 被替换下来的上一任 champion，用于一键回滚 |

* 引用方式：`models:/<名字>@<别名>`（推荐）或 `models:/<名字>/<版本号>`。
* **灰度**：新版本想先验证再上线，用 `register_model --alias staging`，验证后 `promote_model --version N`。
* **从旧流程升级**：升级前注册的版本没有指标标签，所以升级后的第一次注册会被当作不可比较而打 `challenger`。用 `--list` 确认后，直接 `promote_model --version N` 选定 champion 即可，之后的注册就会正常比较。
* **让线上也遵循门禁**：API 默认读本地 `data/` 目录，那里永远是"最新训练的"，不受门禁控制。上线时请让 API 从注册表加载：设置 `TWIN_RANK_MODEL_URI=models:/twin-rank-recommender@champion`（或 `serving.model_uri`）。
* **指标可信度**：晋升比较用的是同一批测试用户。用测试集反复挑选最好的版本，会略微高估效果；上线后的 A/B 测试才是最终依据。`eval_users` 太小时指标波动大，建议 ≥ 10000。

## 7. 调用模型预测

四种方式，都使用注册表里的模型（前提：已设置 `MLFLOW_TRACKING_URI` 并完成注册）。

### 7.1 命令行
```bash
python -m twin_rank.pipelines.predict --user-id <用户ID> --top-k 10
python -m twin_rank.pipelines.predict --user-id <用户ID> --mode rank --explain         # 看推荐理由
python -m twin_rank.pipelines.predict --user-id new_user --history 商品A,商品B,商品C   # 指定历史，给任意/新用户预测
python -m twin_rank.pipelines.predict --model-uri models:/twin-rank-recommender/2        # 指定版本
```
省略 `--user-id` 时取库里的第一个用户。可用的用户 ID 见 `data/features/user_encoder.json`，或访问 API 的 `/users/sample`。

### 7.2 Python
```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/twin-rank-recommender@champion")

request = pd.DataFrame([
    {"user_id": "AXXXX", "top_k": 10},                                   # 按库内历史推荐
    {"user_id": "anyone", "top_k": 5, "history": ["B0AAA", "B0BBB"]},    # 按给定历史推荐
])
print(model.predict(request)[["user_id", "final_rank", "item_id", "title", "rank_score"]])
```
输入列：`user_id`（必填）、`top_k`（默认 10）、`mode`（`recall`/`rank`/`full`，默认 `full`）、`explain`（默认 False）、`history`（可选）。加载这个模型的机器需要装好依赖（torch、faiss-cpu、lightgbm、pandas 等），但**不需要**另外安装 twin_rank 本身，源码已随模型打包。

### 7.3 作为独立 REST 服务（MLflow 自带）
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/twin-rank-recommender@champion" -p 5001 --env-manager local

curl http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d \
 '{"dataframe_split": {"columns": ["user_id","top_k","mode"], "data": [["AXXXX", 10, "full"]]}}'
```
`--env-manager local` 表示用当前环境，不另建虚拟环境。

### 7.4 Twin-Rank 自带的 FastAPI（带缓存、点击反馈、前端 Demo）
```bash
TWIN_RANK_MODEL_URI="models:/twin-rank-recommender@champion" python -m twin_rank.pipelines.serve_api
curl "http://127.0.0.1:8100/recommend?user_id=AXXXX&top_k=10&mode=full&explain=true"
```
| 接口 | 说明 |
|---|---|
| `GET /recommend` | 参数 `user_id`、`top_k`(1–50)、`mode`、`explain` |
| `POST /feedback` | JSON `{"user_id":..., "item_id":...}`，把点击写入历史并清除该用户缓存 |
| `GET /users/{user_id}/history` | 最近行为，参数 `n` |
| `GET /users/sample` | 示例用户 |
| `GET /health` | 健康检查 |

不设置 `TWIN_RANK_MODEL_URI` 时，API 直接读本地 `data/` 目录（开发时更方便）。

## 8. 输出字段说明

| 字段 | 含义 |
|---|---|
| `final_rank` | 最终位次（1 = 最靠前） |
| `item_id` / `title` / `store` / `price` | 商品信息 |
| `source` | `two_tower` 双塔召回；`popular` 热门补充；`pinned` 运营置顶 |
| `recall_rank` / `recall_score` | 召回阶段的名次 / 余弦相似度（越大越相似）。`final_rank` 与 `recall_rank` 的差就是精排+重排带来的位次变化 |
| `rank_score` | 精排分（只有 `rank`/`full` 模式有；`recall` 模式为空）。LightGBM 为排序分，深度模型为点击概率 |
| `ranker`（API 返回） | 本次使用的精排模型类型 |
| `reasons` | 推荐理由（`explain=true` 时），来自 LightGBM 特征贡献 |
| `cold_start` | `true` 表示该用户没有历史，返回的是热门商品 |
| `timings`（API） | 各阶段耗时（毫秒）：用户向量 / 召回 / 精排 / 重排 / 总计 |

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| 下载报 `CERTIFICATE_VERIFY_FAILED` | 已使用 certifi；公司代理设 `SSL_CERT_FILE`；或手动下载文件放进 `data/raw/` |
| macOS 报 `libomp.dylib` 找不到 | `brew install libomp` |
| macOS 上训练/精排极慢或卡住 | 保持默认单线程；不要把 `OMP_NUM_THREADS` 设大 |
| Linux 报 `libgomp.so.1` 找不到 | `sudo apt-get install -y libgomp1` |
| Reflex 首次启动失败（Linux） | `sudo apt-get install -y unzip`，并保证能联网下载 Node/bun |
| 注册模型报错 | 确认 MLflow 服务可达且使用数据库后端（sqlite/MySQL/PostgreSQL） |
| 精排指标没比召回好 | 看 `candidate_hit_rate`；提高 `ranker.max_train_users`；先提升召回 |
| 深度精排比 LightGBM 差 | 通常是样本太少导致过拟合：调大 `max_train_users`、`deep.dropout`，或先用 LightGBM |
| 提示找不到精排模型文件 | `ranker.model_type` 与已训练的类型不一致：重新 `train_ranker`，或改回已训练的类型 |
| 加了重排后 Recall 下降 | 正常，MMR 用相关性换多样性；调大 `mmr_lambda` 或 `max_same_store: 0` 对比 |
| `start.sh` 停在"准备 Python 环境"没动静 | 在安装 PyTorch 等大文件（Linux 默认约数 GB）。用 `du -sh .venv` 看体积是否在增长；国内设置 `PIP_INDEX_URL` 镜像；没有显卡设置 `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` 只装 CPU 版 |
| 想确认环境是否正常 | `python tools/doctor.py` |

## 10. 代码结构

```
configs/config.yaml        全部参数          configs/custom_demo.yaml  合成数据演示配置
src/twin_rank/
  data/        下载、读取、清洗、切分、自定义数据接入
  features/    ID 编码、物品特征、序列样本
  models/      MLP、用户塔/物品塔、双塔模型
  training/    损失(logQ)、指标、评估、训练循环
  retrieval/   Faiss 索引、热门路、多路候选合并、用户塔编码
  ranking/     精排特征、统一接口 base.py、工厂 factory.py、LightGBM(lgbm.py)、
               深度模型 deep/(deepfm / dcn / din)、推荐理由、端到端评估
  rerank/      MMR、品牌打散、黑名单、置顶
  registry/    整套打包、pyfunc 封装、注册、从注册表加载
  serving/     商品目录、行为存储、缓存、推荐主流程、FastAPI
  pipelines/   各步骤的命令行入口（很薄，只做串联）
demo/          Reflex 前端           tools/  启动/停止/自检/示例数据脚本
tests/         单元测试（python -m pytest -q tests）
```
每个源码文件开头都有模块说明，写明它做什么、输入输出和关键设计取舍。
