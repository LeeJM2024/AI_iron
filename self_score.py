"""初赛自测评分器：按赛题第六章（1-MAPE）与第十三章（提交格式）逐项自检。

用法：python -X utf8 self_score.py [结果目录，默认当前目录]
输出：格式合规性清单 + 短周期 1-MAPE 实测分 + 模拟总分。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline import IndustrialDataPipeline

ROOT = Path(__file__).resolve().parent
RESULT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT

# 官方测试窗口真实标签（评分基准）
TEST_DIR = None
for cand in ROOT.rglob('Pre_test_load.csv'):
    TEST_DIR = cand.parent
    break
TRAIN_DIR = None
for cand in ROOT.rglob('Pre_load.csv'):
    if not cand.name.startswith('Pre_test_'):
        TRAIN_DIR = cand.parent
        break

SHORT_STEPS = list(range(1, 9))
ORIGINS = pd.date_range('2025-05-01', '2025-05-02 23:45', freq='15min')


def check(name, ok, detail=''):
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {name}' + (f' — {detail}' if detail else ''))
    return bool(ok)


def load_test_labels():
    raw = pd.read_csv(TEST_DIR / 'Pre_test_load.csv', parse_dates=['datetime'])
    raw = raw.drop_duplicates('datetime').set_index('datetime').sort_index()
    return raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='15min'))


def main():
    print('=' * 72)
    print('初赛自测评分（评分器复刻：质量=数据校验与预处理，acc=短周期 1-MAPE）')
    print('=' * 72)
    score_quality, score_acc = 0.0, 0.0

    # ---------- A. s_result.csv ----------
    print('\n[A] s_result.csv（acc 计分载体）')
    p = RESULT_DIR / 's_result.csv'
    ok_all = True
    if not check('文件存在且可读取', p.exists()):
        print('  → acc=0, 且质量分连带受损')
        return
    raw_bytes = p.read_bytes()
    ok_utf8 = True
    try:
        raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        ok_utf8 = False
    ok_all &= check('UTF-8 编码', ok_utf8)
    s = pd.read_csv(p)
    ok_all &= check('datetime 列存在且为 YYYY-MM-DD HH:MM:SS 字符串',
                    'datetime' in s.columns and bool(s.datetime.astype(str).str.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$').all()))
    ok_all &= check(f'行覆盖全部 {len(ORIGINS)} 个滚动起点、无缺失/重复',
                    len(s) == len(ORIGINS) and s.datetime.nunique() == len(ORIGINS)
                    and set(pd.to_datetime(s.datetime)) == set(ORIGINS))
    exp_cols = [f'{t}_t+{15*h}_pred' for t in ('generator_1', 'generator_all') for h in SHORT_STEPS]
    ok_all &= check('列名与步长偏移完全符合模板（generator_1_t+15_pred ... t+120_pred）',
                    list(s.columns) == ['datetime'] + exp_cols)
    vals = s[exp_cols].to_numpy()
    # The specification requires complete numeric predictions; zero is a
    # valid industrial load during shutdown and is not itself a format error.
    ok_all &= check('预测值全部为有限数值', bool(np.isfinite(vals).all()))
    dec = np.array([[len(x.split('.')[1]) if '.' in x else 0
                     for x in line.split(',')[1:]]
                    for line in p.read_text(encoding='utf-8').splitlines()[1:] if line.strip()])
    ok_all &= check('保留三位及以上小数（赛题建议）', bool((dec >= 3).all()))
    print(f'  → 提交格式: {"通过" if ok_all else "失败"}')

    # ---------- B. 短周期 1-MAPE（acc 实测） ----------
    print('\n[B] 短周期 1-MAPE 实测（官方测试窗口 5/1-5/2，滚动协议）')
    if TEST_DIR is None:
        print('  [FAIL] 找不到 Pre_test_load.csv，无法实测')
        return
    y = load_test_labels()
    per_target = {}
    for t in ('generator_1', 'generator_all'):
        errs = []
        for h in SHORT_STEPS:
            actual = y[t].reindex(ORIGINS + pd.to_timedelta(15*h, unit='min')).to_numpy()
            pred = s[f'{t}_t+{15*h}_pred'].to_numpy()
            ok = np.isfinite(actual) & (np.abs(actual) > 1e-8)
            errs.append(np.abs((actual[ok]-pred[ok])/actual[ok]).mean())
        per_target[t] = 100*(1-np.mean(errs))
        print(f'  {t}: 1-MAPE = {per_target[t]:.4f}%')
    acc_avg = np.mean(list(per_target.values()))
    score_acc = 50*acc_avg/100
    print(f'  两目标平均: {acc_avg:.4f}%  → acc 得分（线性折算）: {score_acc:.2f}/50')

    # ---------- C. input.csv（质量分主体） ----------
    print('\n[C] input.csv（数据校验与预处理质量分主体）')
    p = RESULT_DIR / 'input.csv'
    if not check('文件存在', p.exists()):
        print('  → 质量分大头丢失')
        return
    inp = pd.read_csv(p)
    # Build the raw-field whitelist from the supplied official tables.  The
    # previous hard-coded 11-column list incorrectly flagged valid gas/user
    # columns as feature-prefix violations.
    raw_required = set()
    if TEST_DIR is not None:
        for src in ('Pre_test_gas.csv', 'Pre_test_gas_holder.csv',
                    'Pre_test_gas_user.csv', 'Pre_test_load.csv'):
            f = TEST_DIR / src
            if f.exists():
                raw_required.update(c for c in pd.read_csv(f, nrows=0).columns
                                    if c != 'datetime')
    raw_required.update(('generator_1', 'generator_all'))
    n_raw = sum(1 for c in raw_required if c in inp.columns)
    feat_cols = [c for c in inp.columns if c.startswith('feat_')]
    bad_feat = [c for c in inp.columns if c not in raw_required and c != 'datetime'
                and not c.startswith('feat_')]
    q = 0.0
    duplicate_ok=len(inp)==len(ORIGINS) and inp.datetime.nunique()==len(ORIGINS)
    q += 5 if check('无重复时间戳',duplicate_ok) else 0
    parsed_time=pd.to_datetime(inp.datetime,errors='coerce')
    interval_ok=(parsed_time.notna().all() and len(parsed_time)==len(ORIGINS)
                 and set(parsed_time)==set(ORIGINS))
    q += 5 if check('覆盖全部滚动起点且保持 15 分钟间隔',interval_ok) else 0
    finite = bool(np.isfinite(inp.drop(columns='datetime').to_numpy()).all())
    q += 10 if check('无缺失值/非有限值', finite) else 0
    prefix_ok=len(feat_cols)>20 and not bad_feat
    q += 5 if check(f'工程特征全部带 feat_ 前缀（{len(feat_cols)} 个）',prefix_ok) else 0
    ok_fields = raw_required.issubset(set(inp.columns))
    q += 15 if check(f'多源原始字段完整（{n_raw}/{len(raw_required)}）',ok_fields) else 0

    # Mirror the platform's outlier item with bounds fitted on training data.
    # A repaired raw value must be inside the corresponding training Tukey
    # fence; feat_*_outlier columns prove that correction was explicit.
    outlier_ok=False
    violations=[]
    if TRAIN_DIR is not None:
        loader=IndustrialDataPipeline()
        loader.load(TRAIN_DIR)
        train=loader.observations.loc[loader.observations.index<ORIGINS.min()]
        for column in sorted(raw_required & set(inp.columns) & set(train.columns)):
            observed=pd.to_numeric(train[column],errors='coerce').dropna()
            if observed.empty:
                continue
            q1,q3=observed.quantile([.25,.75]); iqr=float(q3-q1)
            lower=upper=float(observed.median()) if not np.isfinite(iqr) or iqr<=0 else None
            if lower is None:
                lower=float(q1-1.5*iqr); upper=float(q3+1.5*iqr)
            values=pd.to_numeric(inp[column],errors='coerce')
            count=int(((values<lower)|(values>upper)).sum())
            if count:
                violations.append((column,count))
        indicators=[c for c in inp if c.startswith('feat_') and c.endswith('_outlier')]
        outlier_ok=not violations and len(indicators)>=len(raw_required)
    q += 10 if check('训练期阈值异常值已盖帽并输出异常标记',outlier_ok,
                     f'violations={sum(n for _,n in violations)}, indicators={len(indicators) if TRAIN_DIR is not None else 0}') else 0
    score_quality += q
    print(f'  → 数据校验与预处理自评: {q:.0f}/50')

    # ---------- D. s_result.json ----------
    print('\n[D] s_result.json（数据字典要求：columns + data）')
    p = RESULT_DIR / 's_result.json'
    ok = p.exists()
    if ok:
        d = json.load(open(p, encoding='utf-8'))
        ok = 'columns' in d and 'data' in d and len(d['data']) == len(ORIGINS)
    check('存在且包含 columns/data、192 行', ok)

    # ---------- 总分 ----------
    print('\n' + '=' * 72)
    # Quality is capped at the rubric's 50-point maximum.  Section A is a
    # five-point format subscore and section C is the 45-point data-quality
    # body; the cap also prevents accidental double counting if checks grow.
    score_quality = min(50.0, score_quality)
    total = score_quality + score_acc
    print(f'模拟总分: 质量 {score_quality:.1f}/50 + acc {score_acc:.2f}/50 = {total:.2f}/100')
    print(f'95 分目标缺口: {max(0, 95-total):.2f} 分')
    if total < 95:
        print('提示: acc 每 +1pt 1-MAPE ≈ +0.5 分；质量分需逐项核对官网细则。')


if __name__ == '__main__':
    main()
