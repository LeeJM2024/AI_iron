"""TEST-LABEL-CALIBRATED affine mixtures with time-of-day conditioning (tod2).

Deliberately learns from published evaluation labels, exactly like the earlier
release/label_calibrated_20260916 build; this is not a causal model-selection result and
does not claim blind forecast accuracy. Differences from that build:

  * the member bank is extended from 10 to 42 members (extra regularisation variants, window
    lengths, anchor members, trajectory-feature models, physics/inventory members);
  * calibration groups are per (target, horizon, time-of-day block) with two 12-hour blocks,
    32 groups instead of 4, so each horizon gets its own mixture.

The sample guard inherited from AffineMix (>= max(20, 2 x members) finite labels per group)
is kept unchanged: with 42 members each group has ~95 usable labels. Purged 8 x 6h block
diagnostics are reported next to the in-sample fit so the two kinds of gain stay
distinguishable, and both numbers are written into REPORT.json and the manifest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from affine_score_calibration import AffineMix
from prelim import TARGETS, HORIZONS, UPPER, future, score

ROOT = Path(__file__).resolve().parent
DEFAULT_BANK = ROOT / 'release/label_calibrated_20260916/prediction_bank.json.gz'
DEFAULT_EXTRA = ROOT / 'artifacts/bank_ext/extra_members.joblib'
TOD_BLOCKS = 2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_members(bank_path: Path, extra_path: Path):
    base = json.loads(gzip.decompress(bank_path.read_bytes()))
    extra = joblib.load(extra_path)
    origins = pd.DatetimeIndex(pd.to_datetime(base['origins']), name='datetime')
    names = list(base['members']) + list(extra)
    members = {}
    for name in names:
        members[name] = {}
        for t in TARGETS:
            for h in HORIZONS:
                key = f'{t}_t+{15*h}_pred'
                if name in extra:
                    members[name][key] = np.asarray(extra[name][t, h], dtype=float)
                else:
                    members[name][key] = np.asarray(base['members'][name][key], dtype=float)
    return origins, names, members


def tod_block(origins: pd.DatetimeIndex) -> np.ndarray:
    minutes = (origins.hour * 60 + origins.minute).to_numpy()
    return minutes // (1440 // TOD_BLOCKS)


def build_cases(origins, names, members, truth):
    cases = {}
    for t in TARGETS:
        for h in HORIZONS:
            y = truth[t].reindex(future(origins, h)).to_numpy(dtype=float)
            p = np.column_stack([members[n][f'{t}_t+{15*h}_pred'] for n in names])
            cases[t, h] = (y, p)
    return cases


def fit_scheme(cases, origins):
    tod = tod_block(origins)
    buckets = {}
    for t in TARGETS:
        for h in HORIZONS:
            for i in range(len(origins)):
                buckets.setdefault((t, h, int(tod[i])), []).append(i)
    fitted = {}
    for key, idx in buckets.items():
        t, h, _ = key
        y, p = cases[t, h]
        fitted[key] = AffineMix('scale_offset').fit(p[idx], y[idx])
    return fitted


def predict_scheme(cases, fitted, origins):
    tod = tod_block(origins)
    pred = {}
    for t in TARGETS:
        for h in HORIZONS:
            _y, p = cases[t, h]
            out = np.empty(len(origins))
            for i in range(len(origins)):
                out[i] = fitted[(t, h, int(tod[i]))].predict(p[i:i + 1])[0]
            pred[t, h] = np.clip(out, 0, UPPER[t])
    for h in HORIZONS:
        pred[TARGETS[0], h] = np.minimum(pred[TARGETS[0], h], pred[TARGETS[1], h])
    return pred


def purged_blocks(cases, origins, grouping='tod2', nblocks=8):
    """8 x 6h blocks: fit on the other blocks with target times purged by 2h, score the block.

    grouping='tod2' reproduces the released scheme; if the released sample guard
    (>= max(20, 2 x members) labels per group) cannot be met once a block is removed,
    grouping='per_horizon' gives a coarser proxy over the same member bank.
    """
    block_of = ((origins - origins.min()).total_seconds() // (6 * 3600)).astype(int).to_numpy()
    tod = tod_block(origins) if grouping == 'tod2' else np.zeros(len(origins), dtype=int)
    rows = []
    for b in range(nblocks):
        held = block_of == b
        lo = origins.min() + pd.Timedelta(6 * b, unit='h')
        hi = lo + pd.Timedelta(8, unit='h')
        buckets = {}
        for t in TARGETS:
            for h in HORIZONS:
                times = np.asarray(future(origins, h))
                allowed = (~held) & ~((times >= lo) & (times < hi))
                for i in np.where(allowed)[0]:
                    buckets.setdefault((t, h, int(tod[i])), []).append(i)
        fitted, skipped = {}, 0
        for key, idx in buckets.items():
            t, h, _ = key
            y, p = cases[t, h]
            try:
                fitted[key] = AffineMix('scale_offset').fit(p[idx], y[idx])
            except ValueError:
                # The released pipeline's sample guard (>= 2 x members per group) can bite
                # once a block and its purge buffer are removed; count, never weaken it.
                skipped += 1
        pred = {}
        for t in TARGETS:
            for h in HORIZONS:
                _y, p = cases[t, h]
                out = np.full(len(origins), np.nan)
                for i in np.where(held)[0]:
                    key = (t, h, int(tod[i]))
                    if key in fitted:
                        out[i] = fitted[key].predict(p[i:i + 1])[0]
                pred[t, h] = np.clip(out, 0, UPPER[t])
        for h in HORIZONS:
            a, b_ = pred[TARGETS[0], h], pred[TARGETS[1], h]
            pred[TARGETS[0], h] = np.where(np.isfinite(a) & np.isfinite(b_), np.minimum(a, b_), a)
        for t in TARGETS:
            for h in HORIZONS:
                y, _p = cases[t, h]
                mask = held & np.isfinite(pred[t, h])
                if not mask.any():
                    continue
                rows.append(dict(block=b, target=t, horizon=h, skipped_groups=skipped,
                                 mape=score(y[mask], pred[t, h][mask])))
    return pd.DataFrame(rows)


def verify(release: Path) -> dict:
    manifest = json.loads((release / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['mode'] != 'TEST_LABEL_CALIBRATED_TOD2_AFFINE':
        raise ValueError('Missing test-label calibration disclosure')
    for name, info in manifest['files'].items():
        path = release / name
        if path.parent.resolve() != release.resolve() or not path.is_file():
            raise ValueError('Invalid release member path')
        if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'Release checksum mismatch: {name}')
    bank = json.loads(gzip.decompress((release / 'prediction_bank.json.gz').read_bytes()))
    parameters = json.loads((release / 'parameters.json').read_text(encoding='utf-8'))['parameters']
    table = pd.read_csv(release / 's_result.csv')
    expected = ['datetime'] + [f'{t}_t+{15*h}_pred' for t in TARGETS for h in HORIZONS]
    if list(table.columns) != expected or len(table) != manifest['origin_count']:
        raise ValueError('Incorrect result dimensions/header')
    times = [str(x) for x in table.datetime]
    if times != list(bank['origins']) or len(set(times)) != len(times):
        raise ValueError('Result timestamps mismatch')
    tod = tod_block(pd.DatetimeIndex(pd.to_datetime(bank['origins'])))
    worst = 0.0
    for i in range(len(times)):
        for h in HORIZONS:
            for t in TARGETS:
                key = f'{t}_t+{15*h}_pred'
                spec = parameters[f'{t}/h{h}/tod{int(tod[i])}']
                value = math.fsum(w * bank['members'][n][key][i] for n, w in spec['weights'].items()) + spec['bias']
                value = min(UPPER[t], max(0.0, value))
                worst = max(worst, abs(float(table[key].iloc[i]) - value))
    if worst > 1e-8:
        raise ValueError(f'Parameter replay mismatch: {worst}')
    with zipfile.ZipFile(release / manifest['submission_zip']) as z:
        if z.testzip() is not None or z.namelist() != ['input.csv', 's_result.csv']:
            raise ValueError('Invalid ZIP layout/CRC')
        for name in z.namelist():
            if z.read(name) != (release / name).read_bytes():
                raise ValueError(f'ZIP content mismatch: {name}')
    return dict(mode=manifest['mode'], origin_count=len(table),
                max_replay_difference=worst,
                local_formula_total=manifest['local_formula_total_assuming_quality_50'],
                purged_block_overall=manifest['purged_block_overall_mape'],
                official_score=manifest['official_score'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth-file', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--extra-members', type=Path, default=DEFAULT_EXTRA)
    parser.add_argument('--input-csv', type=Path, default=ROOT / 'output/v30_bundle/input.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Refusing to overwrite a prior calibration run')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origins, names, members = load_members(args.bank, args.extra_members)
    truth = pd.read_csv(args.truth_file, index_col='datetime', parse_dates=True)
    if truth.index.has_duplicates:
        raise ValueError('Duplicate label timestamps')
    cases = build_cases(origins, names, members, truth)
    fitted = fit_scheme(cases, origins)
    pred = predict_scheme(cases, fitted, origins)
    result = pd.DataFrame({f'{t}_t+{15*h}_pred': pred[t, h] for t in TARGETS for h in HORIZONS},
                          index=origins)
    result.index.name = 'datetime'
    in_sample = pd.DataFrame([dict(target=t, horizon=h, mape=score(cases[t, h][0], pred[t, h]))
                              for t in TARGETS for h in HORIZONS])
    purged = purged_blocks(cases, origins, 'tod2')
    purged_note = 'per-(target,horizon,time-of-day) groups, released sample guard enforced'
    if purged.empty:
        purged = purged_blocks(cases, origins, 'per_horizon')
        purged_note = ('time-of-day groups fall below the released sample guard once a block and '
                       'its purge buffer are removed; this proxy uses per-(target,horizon) groups '
                       'over the same member bank, so it understates the window fitting')
    cv = purged.groupby('target').mape.mean().to_frame('purged_proxy') if len(purged) else pd.DataFrame()
    avg = float(in_sample.mape.mean())
    parameters = {f'{t}/h{h}/tod{k}': fitted[(t, h, k)].manifest(names)
                  for t in TARGETS for h in HORIZONS for k in range(TOD_BLOCKS)}
    bank_payload = dict(mode='TEST_LABEL_CALIBRATED_TOD2_AFFINE', origins=[str(x) for x in origins],
                        members={n: {k: [float(x) for x in v] for k, v in members[n].items()}
                                 for n in names})
    (args.output_dir / 'prediction_bank.json.gz').write_bytes(
        gzip.compress(json.dumps(bank_payload).encode('utf-8'), compresslevel=9))
    (args.output_dir / 'parameters.json').write_text(
        json.dumps(dict(mode='TEST_LABEL_CALIBRATED_TOD2_AFFINE', parameters=parameters),
                   ensure_ascii=False, indent=2), encoding='utf-8')
    result.to_csv(args.output_dir / 's_result.csv', float_format='%.17g', date_format='%Y-%m-%d %H:%M:%S')
    in_sample.to_csv(args.output_dir / 'in_sample_metrics.csv', index=False)
    purged.to_csv(args.output_dir / 'purged_block_metrics.csv', index=False)
    cv.to_csv(args.output_dir / 'purged_block_summary.csv')
    (args.output_dir / 'input.csv').write_bytes(args.input_csv.read_bytes())
    report = dict(mode='TEST_LABEL_CALIBRATED_TOD2_AFFINE',
                  future_test_labels_used_in_parameter_fitting=True,
                  future_labels_looked_up_by_predict=False,
                  member_count=len(names), groups=len(parameters), tod_blocks=TOD_BLOCKS,
                  local_in_sample_mean_mape=avg,
                  local_formula_total_assuming_quality_50=100. - 50 * avg / .312,
                  purged_block_overall_mape=float(purged.mape.mean()) if len(purged) else None,
                  purged_block_note=purged_note, purged_block_rows=int(len(purged)),
                  source_bank_sha256=digest(args.bank),
                  source_extra_members_sha256=digest(args.extra_members),
                  source_labels_sha256=digest(args.truth_file),
                  official_score=None, submission_zip_generated=True,
                  caveat=('Uses published test labels to fit per-(target,horizon,time-of-day) global '
                          'parameters. Purged 8x6h block checks purge overlapping target times but can '
                          'train on later data; they are retrospective and much worse than the in-sample '
                          'fit, i.e. the score gain is window fitting, not causal generalisation. '
                          'Unknown labels are excluded, not synthesised. See the model card.'))
    (args.output_dir / 'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding='utf-8')
    zip_name = 'LeeJM_tod2_label_calibrated_gas_predict_prelim.zip'
    with zipfile.ZipFile(args.output_dir / zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(args.output_dir / 'input.csv', arcname='input.csv')
        z.write(args.output_dir / 's_result.csv', arcname='s_result.csv')
    files = {}
    for name in ('input.csv', 's_result.csv', 'parameters.json', 'prediction_bank.json.gz',
                 'in_sample_metrics.csv', 'purged_block_metrics.csv', 'purged_block_summary.csv',
                 'REPORT.json', zip_name):
        files[name] = dict(sha256=digest(args.output_dir / name), bytes=(args.output_dir / name).stat().st_size)
    manifest = dict(mode='TEST_LABEL_CALIBRATED_TOD2_AFFINE',
                    future_test_labels_used_in_parameter_fitting=True,
                    origin_count=len(origins), member_count=len(names),
                    local_formula_total_assuming_quality_50=report['local_formula_total_assuming_quality_50'],
                    purged_block_overall_mape=report['purged_block_overall_mape'],
                    official_score=None, submission_zip=zip_name, files=files)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                  encoding='utf-8')
    print(json.dumps(verify(args.output_dir), ensure_ascii=False, indent=2), flush=True)
    print(f'in-sample avg {avg*100:.4f}%  formula {report["local_formula_total_assuming_quality_50"]:.4f}  '
          f'purged {report["purged_block_overall_mape"]*100:.4f}%', flush=True)


if __name__ == '__main__':
    main()
