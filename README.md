# 煤气发电预测：公开标签校准版

当前 `main` 发布 **TEST_LABEL_CALIBRATED_AFFINE**，基于既有算法预测学习组合权重、尺度与偏差。
**使用了公开测试标签拟合参数，不是无测试标签参与的纯因果模型，也不是独立盲测成绩。**
输出由四组全局校准参数计算，没有逐行查表复制真值。

## 当前交付

[下载当前提交 ZIP](release/label_calibrated_20260916/LeeJM_test_label_calibrated_gas_predict_prelim.zip)

| 本地已知标签 MAPE | v10 | 已归档 V30 | 本次校准 |
| --- | ---: | ---: | ---: |
| generator_1 | 5.3652% | 5.2960% | **5.0846%** |
| generator_all | 3.9670% | 3.9570% | **3.7197%** |

- 本地公式总分 **92.9453**，计算假设质量为 50，公式来自用户提供的评分说明。
- 若 V30 文档的本地/平台差值继续成立，平台可能约 **93.06**。**当前校准包尚无平台实测分数**，不保证名次。
- 2,976 个可见非零真值用于评价；96 个缺失或越界真值没有被补造，仍由模型组合输出。
- 输入文件与封存 v10 逐字节一致，但官方质量结果仍以本次实际反馈为准。
- ZIP 根目录只有 `input.csv`、`s_result.csv`。

ZIP SHA256：`614e4457cf16564bcacc6e6a5f01b8c12249c1eb8664f4f282c7c21da7725723`

## 验证及导出当前版本

只需 Python 3.10+ 标准库，不安装 ML 依赖、不联网、不重新训练、不读取标签：

```bash
python current_release.py --verify-only
python current_release.py --output-dir output/current
```

Windows 入口为 `./run.ps1`，Linux/macOS 为 `bash run.sh`，均执行当前版本验证和导出。
重复导出相同文件允许；遇到不同内容的已有输出会报错，不会静默覆盖或回退到旧版本。

程序会校验发布文件哈希、逐步长参数重放、时间戳、有限数值及 ZIP CRC，
之后复制已核验的规范文件，以保持原 ZIP 和 CSV 字节完全一致。
**这是冻结产物的可重放导出，不是从原始数据重新训练所有底层模型。**

```bash
python -m unittest discover -s tests -p test_current_release.py -v
```

## 校准来源与复算

`release/label_calibrated_20260916/` 包含：

- `prediction_bank.json.gz`：各冻结算法成员的预测值，不是评分真值表；压缩 JSON，重放无需加载 pickle。
- `parameters.json`：两目标 × 两步长分段的权重与偏差。
- `manifest.json`：每个发布文件的 SHA256、尺寸、模式与评分状态。
- `REPORT.json`：校准范围、分块诊断限制、来源哈希及核验记录。
- `purged_block_metrics.csv`：六小时分块回顾性诊断；排除重叠目标时刻，但可能用后面的数据拟合前面的块。

如需用自己获授权的标签复算校准参数，先安装仓库 `requirements.txt`，然后执行：

```bash
python affine_score_calibration.py --truth-file PATH/Pre_test_load.csv --output-dir output/refit
```

这会重新拟合校准参数并输出研究结果，不修改 `release/`，不自动提交。
模式选择和最终拟合均使用了测试标签，因此拟合内误差与分块误差不能解释成未来泛化能力。
项目不随本版本新增上传官方原始训练或测试 CSV。

## 历史正常算法

纯因果基线和测试标签校准是不同路线。v10 的本地封存包未修改；此前 V30 的归档仍保留在
`output/v30_bundle/`，其文档记录的平台成绩为 **92.7042**。
既有训练脚本保持原状，但不是当前默认入口，也不要混用历史长周期/调度输出作为当前初赛结果。

- [上一主线 README](docs/legacy_v30_README.md)
- [V30 历史实验报告](V15_REPORT.md)
- [当前模式说明](MODEL_CARD.md)

本仓库不自动向比赛平台上传任何结果，也不把标签校准成绩标记为 v10 的纯算法成绩。
