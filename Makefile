.PHONY: help doctor pipeline test clean
help:            ## 显示所有命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

doctor:          ## 环境自检（依赖、GPU、MLflow 连通性）
	python tools/doctor.py
download:        ## 步骤1 下载 Amazon 数据（自定义数据会跳过）
	python -m twin_rank.pipelines.download_data
preprocess:      ## 步骤2 清洗 + 按时间切分
	python -m twin_rank.pipelines.preprocess_data
features:        ## 步骤3 特征工程
	python -m twin_rank.pipelines.build_features
train:           ## 步骤4 训练双塔召回模型
	python -m twin_rank.pipelines.train_two_tower
index:           ## 步骤5 构建 Faiss 向量索引
	python -m twin_rank.pipelines.build_index
ranker:          ## 步骤6 训练精排模型（ranker.model_type 决定用哪个）
	python -m twin_rank.pipelines.train_ranker
compare-rankers: ## 依次训练 lightgbm/deepfm/dcn/din 并在 MLflow 对比
	for t in lightgbm deepfm dcn din; do TWIN_RANK_RANKER=$$t python -m twin_rank.pipelines.train_ranker; done
register:        ## 注册到 MLflow（指标超过 champion 才晋升）
	python -m twin_rank.pipelines.register_model
models:          ## 查看注册表各版本、别名、指标
	python -m twin_rank.pipelines.promote_model --list
predict:         ## 用注册表里的 champion 做一次推荐
	python -m twin_rank.pipelines.predict
pipeline: download preprocess features train index ranker register  ## 一次跑完 1-6 + 注册
serve:           ## 启动推荐 API（默认 8100 端口）
	python -m twin_rank.pipelines.serve_api
app:             ## 启动 Reflex Web Demo（默认 3000 端口）
	cd demo && reflex run
test:            ## 运行单元测试
	python -m pytest -q
clean:           ## 清理缓存（不会删除 data/ 下的数据与模型）
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache logs .api.pid
