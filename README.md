# 煤气发电预测与调度

## 共享步长候选（第四版）

`./run_pooled.ps1` 使用 `artifacts/pooled_v4_selected/selection.json` 的历史验证选择，
输出到独立的 `output/pooled_v4/`。它沿用第三版清洗，通过共享 8 个步长的 LightGBM
学习相对负荷变化；只对在训练内留一折评估中达到收益阈值的目标启用。
`./run_pooled.ps1 -Develop` 可完整重做历史实验。

`run_prelim.py --warm-start-model output/stable_v3/prelim_model.joblib` 可复用第三版
相同参数的成员，代码会核对训练数据哈希、截止时间、预处理配置与训练特征。
只加载自己生成的可信 joblib 文件。新增模型仍正常训练，所有输出重新核验。

## 第二轮：质量项与输入预处理优化

针对平台反馈 `quality=40/50`、`out=0`、`invalid_col=0`，新增
`run_stable.ps1`（内部使用 `run_prelim.py --selection artifacts/stable_v3_selected/selection.json`）。
结果独立输出到 `output/stable_v3/`，不覆盖 82.5 分的上一包。

此分支仅根据训练末期停用近期无变化的传感器，采用因果的非负荷传感器分位数截断，
对变化量及周期特征使用有界非负编码，并同步重训模型。不得只更改提交 CSV 而继续用
旧输入生成预测。完整实验可用 `./run_stable.ps1 -Develop` 重做。

本地诊断不等于官方评分，不能根据“常数列=0”便声称质量已满分。
第二轮实测结果和局限见 `STABLE_V3_REPORT.md`。

## 2026-09-12 新增：初赛短周期专用入口

本次优化代码与新结果独立保存，不覆盖下面旧流程的预置结果。**初赛请用
`run_prelim.ps1` / `run_prelim.py`，提交它实际生成的两文件 ZIP，不要提交仓库根目录的旧 CSV。**

```powershell
.\run_prelim.ps1
```

默认读取本仓库上一级的官方训练、测试数据目录。也可显式指定：

```powershell
python -X utf8 run_prelim.py --train-dir "训练目录" --test-dir "测试目录" --output-dir output/prelim_v2
```

Linux：`bash run_prelim.sh "训练目录" "测试目录"`。先安装 `requirements.txt`。
当前本机在独立 `.venv` 中安装 LightGBM/CatBoost，不更改上一级旧代码。

新入口只处理初赛 8 个步长，不计算无关的 96 步调度。按目标及预测时距选择
Ridge/鲁棒线性 ARX、LightGBM、CatBoost 与持续值候选。历史折外预测拟合非负
MAPE 集成权重，最后一个 4 月窗口不参与权重拟合；5 月不参与参数选择。
支持仅使用已经到期的历史预测误差进行滚动偏差修正。

输出 `output/prelim_v2/LeeJM_gas_predict_prelim.zip`，ZIP 根目录只有
`input.csv`、`s_result.csv`。同目录的模型、历史选择、训练标签截止审计、质量报告和
测试误差明细用于复现，不混入提交 ZIP。异常退出不会返回预置预测冒充成功。
`quality_audit.json` 不是官方评分器，不会把物理上有效的停机、零流量自动称作异常。

完整重做训练内候选实验：`.\run_prelim.ps1 -Develop`。单独运行测试：
`python -m pytest tests/test_prelim.py -q`。新模型与旧实现的实测比较见
`PRELIM_IMPROVEMENT.md`。

## 提交包内容与运行方式（评测入口）

本包为自足式提交：

- **评分所需文件已预置在包根目录**：`s_result.csv`（短周期，datetime + generator_1/generator_all × t+15..t+120 共 17 列）、`l_result.csv`（长周期 193 列）、`input.csv`（datetime + 原始字段 + feat_ 前缀工程特征）、`opt_result.csv`（opt_ 前缀调度结果）、`result.csv`（短周期结果副本，兼容"仅含 result.csv"的提交规范）、`s_result.json`（数据字典要求的 JSON，含 columns/data 字段）。文件均为 UTF-8 编码，字段与变量名全英文，列名与赛题模板一致。
- **评测入口**：`bash run.sh`（Linux）或 `python run.ps1` / `.
un.ps1`（Windows）。
  - 脚本会先确保上述结果文件就位（包内 `results_prebaked/` 兜底副本）；
  - 随后自动在包目录及常见路径（`data/`、`/data`、`/dataset` 等）定位赛题数据包（含 `Pre_load.csv` 的目录，或用环境变量 `DATA_DIR_OVERRIDE` 直接指定）；
  - 找到数据则运行完整管线（冻结模型训练 + 滚动预测 + 调度 + 校验，约 9-15 分钟），**成功后用新鲜结果覆盖根目录文件**；任何失败都保留预置结果，不影响评分文件存在性。
- 依赖见 `requirements.txt`（numpy/pandas/scipy/scikit-learn/lightgbm/catboost/openpyxl/joblib）；入口脚本不依赖网络。
- 冻结的模型选择位于 `artifacts/development/selection.json`（含配置与集成权重），随包分发，无需重跑历史实验。


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
