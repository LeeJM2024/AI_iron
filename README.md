# 煤气发电预测与调度

已适配当前初赛训练、测试数据包。正式流程使用按时间验证选择的 LightGBM / CatBoost 残差集成，输出完整 8 步及 96 步预测，并使用 SciPy HiGHS 求解 96 步整数机组调度。

实测结果见 `VALIDATION_REPORT.md`。输出位于 `output/official/`，旧的 `output_initial/` 不代表本次结果。预测分数与调度仿真收益分开评价，不承诺排行榜名次。

## 安装与运行

赛题要求 Python 3.8–3.10，本版参赛环境请选择 **Python 3.10**。当前机器实际验证环境为 Python 3.13.5，尚未在官方 Python 3.10 容器验证。依赖范围兼容 3.10，实际版本记录在 `validation.json`。

```powershell
cd D:\learning\ai_steel
python -m pip install -r requirements.txt
.\run.ps1
```

Linux / Bash：`bash run.sh`。脚本在缺少模型选择文件时先运行历史实验，此后复用冻结的选择结果，再训练最终模型、预测、调度并校验输出。每次完整运行仍会重新训练模型，失败返回非零退出码。

分开执行便于复现：

```powershell
python -X utf8 experiment.py
python -X utf8 main.py --official --output-dir output/official
python -X utf8 validate_outputs.py --output-dir output/official
python -m unittest discover -s tests -p test_invariants.py -v
$env:PYTHONPATH=(Get-Location).Path
python tests/smoke_test.py
```

`validate_outputs.py` 用于单独复查已生成结果，正式入口也会自动调用。`main.py --official` 使用 `artifacts/development/selection.json`；需要重新选择模型时先运行 `experiment.py`。不要使用测试评分反复挑选参数。

## 数据与时间边界

无需改名原始文件。程序递归找到唯一的 `Pre_load.csv`，从同级目录读取 `Pre_gas.csv`、`Pre_gas_holder.csv`、`Pre_gas_user.csv`、`price.xlsx`。测试使用 `Pre_test_*.csv`。多个同名包会报错，应通过 `--input-dir` 指向唯一包。

1. 多表按时间排序、去重，重采样为 15 分钟。区间为 `(t-15min,t]`，不会把 `t` 之后的数据放进 `t` 的特征。
2. 原始标签独立保存。只对特征前向填充，首次缺测用零及缺失标记表示；不把填充值当作真实标签，不通过平滑去除真实停机或工况突变。
3. 工程字段带 `feat_`。包含过去分位数盖帽视图、多尺度滞后/均值/波动、工况变化、逐煤气产耗代理量、单柜裕度及目标时刻日历。未知单位的产耗差只作为代理特征，不能解释为可调度气量。
4. 4 月 7 日、18 日两个历史窗口选择集成权重，4 月 27 日是未参与选择的历史验证窗口。开发时长周期抽样验证第 16、24、48、96 步，正式预测覆盖全部 96 步。
5. 各 horizon 剔除标签时刻越过训练截止点的样本。最终模型只使用 5 月测试开始之前的标签。测试按滚动协议使用截至起点 `t` 的已观测数据，预测 `t+15min` 至 `t+24h`。

评估假设每隔 15 分钟能得到最新实际负荷。若官方协议从单一时刻一次性预测整个测试月，期间不允许使用新观测，必须按该协议重新评估，本报告分数不能直接沿用。

## 模型

每个目标、每个步长直接学习“未来负荷减当前负荷”，避免递归误差累积。LightGBM 与 CatBoost 使用逆负荷权重的 MAE 目标逼近 MAPE，权重下限仅由训练样本确定。

候选还包括当前值持续、指数平滑、目标时刻前一天值。用历史折外预测学习非负且和为 1 的集成权重，按目标及步长区间选择。远期没有稳定收益的复杂成员允许降权至零，零权重模型跳过训练。最终预测限制在额定上界内，且 `generator_1 <= generator_all`。

这是针对当前工业表格时序数据验证的集成方案，不把算法名称当作效果证明。样本覆盖不同工况的程度仍限制远期质量。

## 调度及适用边界

全 96 步均使用 MILP，不忽略机组整数约束。四台 50MW、两台 120MW 机组采用整数在线台数，在线负荷为额定的 60%–100%，允许停机。包含爬坡约束、变化及启机惩罚、分时电价收入与放散惩罚。

三类煤气分别守恒，不能互借体积。只有观测到的高炉煤气柜允许跨时段存储。根据随包字典，1 号柜为 20 万 m³，2 号柜为 30 万 m³；本数据中 1 号柜全空，使用 2 号柜实测库存及 15%–90% 安全区间。末端库存至少回到初始值，防止透支库存虚增收益。

字典未明确煤气流量单位，当前假设发电用气为 m³/h。按历史发电用气及气柜库存变化估计优先用户用气后的可用资源，不把数量级不一致的转炉产量直接相减。气电转换采用历史非负回归估计，资源预测采用最近观测的稳健水平。

调度结果是在给定资源预测边界下的约束验证与收益仿真。未测柜库存、真实未来供需、机组初始在线状态、最小开停机时间、动态热值及现场规程仍需补齐。10%/min 的 15 分钟爬坡限制较宽，单步平滑主要由惩罚引导，并非保证额外的硬平滑阈值。

限时求解接受经全部约束复核的整数可行解，并记录 MIP gap；无可行解明确失败。缺失电价或初始库存越界也会失败，不伪造电价、应急煤气或安全库存。

## 输出

所有正式结果位于 `output/official/`：

| 文件 | 含义 |
| --- | --- |
| `s_result.csv` | 起点及两目标 × 8 步，共 17 列 |
| `l_result.csv` | 起点及两目标 × 96 步，共 193 列 |
| `opt_result.csv` | 96 行调度，三类消耗字段带 `opt_` |
| `input.csv` | 对齐起点的原始字段及公共 `feat_` 特征 |
| `s_result.json` | 字典要求的 `columns` / `data` 结构 |
| `forecast_model.joblib` | 已训练模型，仅加载可信文件 |
| `training_audit.csv` | 192 个目标/步长的最后特征、标签及截止时间 |
| `test_summary.csv` | 测试 MAPE 与持续值基线 |
| `test_metrics_by_horizon.csv` | 每步评分及缺失/零标签计数 |
| `test_metrics_by_day.csv` | 按天诊断工况误差 |
| `dispatch_audit.csv` | 在线台数、功率、气柜和放散路径 |
| `dispatch_inputs.csv` | 资源预测、转换系数和电价，便于复算 |
| `run_metadata.json` | 耗时、求解状态、收益及假设 |
| `validation.json` | 独立核验、评分、版本与数据哈希 |

`t+15` 至 `t+1440` 的偏移量单位为分钟，遵循原始 Markdown。当前同时交付 CSV 与短周期 JSON，最终提交以主办方实际模板为准。超出测试数据末尾的目标仍完整预测，但没有真实标签的格子不纳入评分。

## 模块

`pipeline.py`：多表加载、电价及兼容接口；`forecasting.py`：因果特征和残差集成；`experiment.py`：历史模型选择；`dispatch.py`：整数调度；`run_official.py`：冻结模型的正式测试；`validate_outputs.py`：从导出文件独立复核。

`main.py` 保留平铺 CSV 兼容入口，`--fast` 仅供旧入口冒烟验证，不能替代正式方案。
