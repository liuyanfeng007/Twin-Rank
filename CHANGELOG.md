# 更新日志

## 1.0.0
* 整理项目：重写 README，新增 ARCHITECTURE / PROJECT_SUMMARY / CHANGELOG，补全 `.gitignore`、`.gitattributes`，Makefile 增加 `help`、`doctor`、`clean`，统一版本号。

## 0.6
* **GPU 兼容检测**：`device: auto` 先确认当前 PyTorch 是否支持这块 GPU（如新版 PyTorch 不再支持 Tesla P100），不支持则回退 CPU 并提示修复命令。
* 启动脚本：依赖安装显示进度、检测软件源连通性、`DEBUG=1` 调试开关、支持 `PIP_INDEX_URL` / `TORCH_INDEX_URL`。

## 0.5
* **注册质量门禁**：新版本指标超过 champion 才晋升；否则打 `challenger`；口径不同不自动晋升。
* 训练证据文件 + 文件指纹，保证指标对应被注册的模型；`promote_model` 支持列表 / 手动晋升 / 回滚。

## 0.4
* **精排可切换**：LightGBM / DeepFM / DCN-V2 / DIN，统一 `Ranker` 接口，`TWIN_RANK_RANKER` 环境变量，`make compare-rankers`。

## 0.3
* MLflow 模型注册（整套系统打包为 pyfunc）与按注册表预测；自定义数据接入；使用文档；环境自检 `doctor`；macOS / Linux 兼容。

## 0.2
* 向量索引与多路召回、LightGBM 精排、MMR 重排、FastAPI 服务、Reflex 演示、一键启动脚本。
* 修复：macOS 下 libomp 冲突导致精排卡死（默认单线程）；精排训练排除无正样本的用户。

## 0.1
* 数据获取、预处理、特征工程、双塔召回训练（MLflow）。
* 修复：macOS 下载 SSL 证书；MLflow 3 的 pytorch 模型记录格式。
