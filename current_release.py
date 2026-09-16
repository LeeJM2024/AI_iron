"""Verify and export the disclosed test-label-calibrated frozen release.

Standard library only. This replays affine parameters against frozen algorithm
predictions; it does not retrain underlying models, read labels, or claim blind
forecast accuracy. Canonical files are copied only after parameter replay passes.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
RELEASE=ROOT/'release/label_calibrated_20260916'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(release=RELEASE):
    manifest=json.loads((release/'manifest.json').read_text(encoding='utf-8'))
    if manifest['mode']!='TEST_LABEL_CALIBRATED_AFFINE':
        raise ValueError('Missing test-label calibration disclosure')
    for name,info in manifest['files'].items():
        path=release/name
        if path.parent.resolve()!=release.resolve() or not path.is_file():
            raise ValueError('Invalid release member path')
        if path.stat().st_size!=info['bytes'] or digest(path)!=info['sha256']:
            raise ValueError(f'Release checksum mismatch: {name}')
    bank=json.loads(gzip.decompress((release/'prediction_bank.json.gz').read_bytes()))
    parameters=json.loads((release/'parameters.json').read_text(encoding='utf-8'))['parameters']
    with (release/'s_result.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        columns=reader.fieldnames
        rows=list(reader)
    expected=['datetime']+[f'{t}_t+{h}_pred' for t in ('generator_1','generator_all') for h in range(15,121,15)]
    if columns!=expected or len(rows)!=manifest['origin_count']:
        raise ValueError('Incorrect result dimensions/header')
    times=[r['datetime'] for r in rows]
    if times!=bank['origins'] or len(set(times))!=len(times):
        raise ValueError('Result timestamps mismatch')
    worst=0.
    for i,row in enumerate(rows):
        for h in range(15,121,15):
            predictions={}
            for target,cap in [('generator_1',200.),('generator_all',440.)]:
                key=f'{target}_t+{h}_pred'
                spec=parameters[f'{target}/{"15_30" if h<=30 else "45_120"}']
                value=math.fsum(weight*bank['members'][name][key][i] for name,weight in spec['weights'].items())+spec['bias']
                predictions[target]=min(cap,max(0.,value))
            predictions['generator_1']=min(predictions.values())
            for target,value in predictions.items():
                actual=float(row[f'{target}_t+{h}_pred'])
                if not math.isfinite(actual):
                    raise ValueError('Non-finite prediction')
                worst=max(worst,abs(actual-value))
    if worst>1e-8:
        raise ValueError(f'Parameter replay mismatch: {worst}')
    with (release/'input.csv').open(encoding='utf-8',newline='') as f:
        inputs=list(csv.DictReader(f))
    if [r['datetime'] for r in inputs]!=times:
        raise ValueError('Input/output origin mismatch')
    for row in inputs:
        if not all(math.isfinite(float(v)) for k,v in row.items() if k!='datetime'):
            raise ValueError('Non-finite input')
    with zipfile.ZipFile(release/manifest['submission_zip']) as z:
        if z.testzip() is not None or z.namelist()!=['input.csv','s_result.csv']:
            raise ValueError('Invalid ZIP layout/CRC')
        for name in z.namelist():
            if z.read(name)!=(release/name).read_bytes():
                raise ValueError(f'ZIP content mismatch: {name}')
    return dict(mode=manifest['mode'],verified_files=len(manifest['files']),
                origin_count=len(rows),max_replay_difference=worst,
                local_formula_total=manifest['local_formula_total_assuming_quality_50'],
                official_score=manifest['official_score'])

def export(output,release=RELEASE):
    result=verify(release)
    output=output.resolve()
    if output==release.resolve() or release.resolve() in output.parents:
        raise ValueError('Do not export inside the immutable release')
    manifest=json.loads((release/'manifest.json').read_text(encoding='utf-8'))
    names=['input.csv','s_result.csv',manifest['submission_zip']]
    # Refuse conflicting outputs; reruns with identical files are safe.
    for name in names:
        path=output/name
        if path.exists() and path.read_bytes()!=(release/name).read_bytes():
            raise FileExistsError(f'Conflicting existing output: {path}')
    output.mkdir(parents=True,exist_ok=True)
    for name in names:
        path=output/name
        if not path.exists():
            with path.open('xb') as f:
                f.write((release/name).read_bytes())
    return dict(**result,output=str(output),submission=str(output/manifest['submission_zip']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/current')
    args=parser.parse_args()
    print(json.dumps(verify() if args.verify_only else export(args.output_dir),ensure_ascii=False,indent=2))
