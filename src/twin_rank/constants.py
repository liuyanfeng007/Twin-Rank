"""全项目共用的常量。"""

# 推荐模式：
#   recall = 仅双塔(+热门)召回排序
#   rank   = 召回 + LightGBM 精排
#   full   = 召回 + 精排 + MMR 重排 / 业务规则（线上默认）
MODES = ("recall", "rank", "full")

# 可选的精排模型（配置项 ranker.model_type，或环境变量 TWIN_RANK_RANKER 临时覆盖）：
#   lightgbm  树模型，小数据稳定、训练快、可解释      deepfm  FM 低阶交叉 + DNN 高阶交叉
#   dcn       DCN-V2，显式有界阶交叉 + DNN            din     目标注意力，按候选商品选择性关注用户历史
RANKER_TYPES = ("lightgbm", "deepfm", "dcn", "din")
