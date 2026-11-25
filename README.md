# Twin-Rank

**双塔向量召回 + 可切换精排（LightGBM / DeepFM / DCN / DIN）+ MMR 重排** 的端到端推荐系统。
覆盖从数据到上线的完整链路：数据处理 → 训练 → 向量索引 → 精排 → 重排 → 模型注册（质量门禁）→ 在线服务 → Web 演示。

```
 数据 ─► 清洗/时间切分 ─► 特征 ─► 双塔训练 ─► Faiss 索引 ─► 多路召回(~100) ─► 精排 ─► MMR 重排 ─► Top-K
                                     │                                      ▲
                                  MLflow 记录 ─► 整套打包注册 ─► 质量门禁 ─► champion ─► FastAPI ─► Reflex Demo
```

## 特性

| 模块 | 内容 |
|---|---|
| 召回 | 双塔（用户 ID + 行为序列 / 商品 ID + 类目品牌价格热度），batch 内负采样 + logQ 修正；Faiss 精确或 HNSW 检索；双塔 + 热门多路合并 |
| 精排 | 统一接口下可配置切换 **LightGBM / DeepFM / DCN-V2 / DIN**，同一份候选、同一套特征，公平对比 |
| 重排 | MMR 多样性、品牌上限、黑名单、运营置顶 |
| MLOps | MLflow 记录实验；**整套系统**打包为一个 pyfunc 模型注册；**质量门禁**：指标超过 champion 才晋升；一键回滚 |
| 服务 | FastAPI（缓存 Redis/内存、点击反馈、推荐理由、各阶段耗时）；Reflex 前端并排对比召回 vs 最终结果 |
| 数据 | Amazon Reviews 2023，或**自定义数据**（列名映射即可，商品表可选） |
| 工程 | macOS / Linux / Windows；环境自检 `doctor`；GPU 不兼容自动回退 CPU；30 个单元测试 |

## 快速开始

```bash
tools/start.sh             # macOS / Linux 一键启动（Windows: tools\start.bat）
```
脚本会：建虚拟环境 → 装依赖 → 环境自检 → 跑完整流水线并注册模型 → 启动 API（8100）→ 启动 Demo（http://localhost:3000）。
首次运行需要下载数据并训练，时间较长；之后会跳过。强制重训加 `--retrain`，停止用 `tools/stop.sh`。

不联网、几分钟体验完整流程（合成数据）：
```bash
pip install -e ".[demo,dev]"
python tools/make_demo_custom_data.py
C=configs/custom_demo.yaml
for s in preprocess_data build_features train_two_tower build_index train_ranker register_model; do
  python -m twin_rank.pipelines.$s --config $C; done
```

## 常用命令（`make help` 查看全部）

```bash
make pipeline        # 步骤 1-6 + 注册
make compare-rankers # 一次训练四种精排并在 MLflow 对比
make models          # 查看注册表各版本 / 别名 / 指标
make predict         # 用 champion 做一次推荐
make serve           # API  http://127.0.0.1:8100/docs
make app             # Web Demo
make test            # 单元测试
```
切换精排模型：改 `configs/config.yaml` 的 `ranker.model_type`，或临时 `TWIN_RANK_RANKER=din make ranker`。

## 项目结构

```
Twin-Rank/
├── configs/            config.yaml（全部参数，带注释）、custom_demo.yaml（合成数据演示配置）
├── src/twin_rank/
│   ├── data/           下载、读取、清洗、时间切分、自定义数据接入
│   ├── features/       ID 编码、物品特征、序列样本
│   ├── models/         用户塔 / 物品塔 / 双塔
│   ├── training/       损失(logQ)、指标、评估、训练循环
│   ├── retrieval/      Faiss 索引、热门路、多路候选、用户塔编码
│   ├── ranking/        精排接口、LightGBM、deep/(DeepFM·DCN·DIN)、特征、推荐理由、端到端评估
│   ├── rerank/         MMR、品牌打散、黑名单、置顶
│   ├── registry/       整套打包、pyfunc 封装、质量门禁、别名与版本管理
│   ├── serving/        商品目录、行为存储、缓存、推荐主流程、FastAPI
│   ├── pipelines/      各步骤命令行入口（很薄，只做串联）
│   └── utils/          设备选择与 GPU 兼容检测、MLflow 工具、文件指纹
├── demo/               Reflex 前端
├── tools/              start/stop 脚本（sh + bat）、环境自检、示例数据生成
├── tests/              单元测试
└── docs/               USAGE.md、ARCHITECTURE.md、PROJECT_SUMMARY.md
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | **使用手册**：全部参数详解、换成自己的数据、精排模型怎么选、模型注册与版本管理、四种预测调用方式、常见问题 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **设计说明**：数据流与产物、关键设计决策及原因、训练/服务一致性、评估口径、已知局限 |
| [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | **项目总结**：亮点、技术栈、示例结果、简历写法、演示脚本、常见追问 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |

## 环境要求

Python 3.9–3.12；macOS / Linux / Windows；GPU 可选（自动检测，不兼容时回退 CPU）；需要一个 MLflow 服务（数据库后端）用于注册模型。
遇到环境问题先运行 `python tools/doctor.py`，常见问题见 [USAGE.md 第 9 节](docs/USAGE.md)。
