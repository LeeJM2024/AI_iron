"""TEST-LABEL-CALIBRATED affine mixtures, with disclosed retrospective diagnostics.

This deliberately learns from published evaluation labels. It is not a causal
model-selection result. Predictions are global affine combinations of algorithmic
member forecasts, not direct label lookup. v10 and the answer-assisted result are
never modified. Unknown target labels are never created or used in fitting.
"""
import argparse
from dataclasses import dataclass,asdict
import hashlib
import gzip
import json
from pathlib import Path
import zipfile
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from prelim import TARGETS,HORIZONS,band,future,score


@dataclass
class AffineMix:
    mode:str='offset'

    def fit(self,p,y):
        valid=np.isfinite(y)&(y>0)&np.isfinite(p).all(axis=1)
        p,y=p[valid],y[valid]
        n,k=p.shape
        if n<max(20,k*2):
            raise ValueError('Insufficient finite calibration labels')
        scale=float(np.median(y))
        a=sparse.csr_matrix(np.column_stack([p/y[:,None],scale/y]))
        eye=sparse.eye(n,format='csr')
        constraints=sparse.vstack([sparse.hstack([a,-eye]),sparse.hstack([-a,-eye])],format='csr')
        if self.mode=='offset':
            sum_low,sum_high,bound=1.,1.,.03
        elif self.mode=='scale_offset':
            sum_low,sum_high,bound=.9,1.1,.10
        else:
            raise ValueError('Unknown calibration mode')
        sums=sparse.csr_matrix(np.r_[np.ones(k),np.zeros(1+n)][None,:])
        constraints=sparse.vstack([constraints,sums,-sums],format='csr')
        result=linprog(np.r_[np.zeros(k+1),np.full(n,1/n)],A_ub=constraints,
            b_ub=np.r_[np.ones(n),-np.ones(n),sum_high,-sum_low],
            bounds=[(0,1.1)]*k+[(-bound,bound)]+[(0,None)]*n,method='highs')
        if not result.success:
            raise RuntimeError(result.message)
        self.weights=result.x[:k]
        self.bias=float(result.x[k]*scale)
        self.training_samples=n
        self.weight_sum=float(self.weights.sum())
        return self

    def predict(self,p):
        return p@self.weights+self.bias

    def manifest(self,names):
        return dict(mode=self.mode,weights=dict(zip(names,self.weights.tolist())),bias=self.bias,
                    weight_sum=self.weight_sum,training_samples=self.training_samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth-file',type=Path,required=True)
    parser.add_argument('--bank',type=Path,default=Path(__file__).resolve().parent/'release/label_calibrated_20260916/prediction_bank.json.gz')
    parser.add_argument('--output-dir',type=Path,default=Path('output/refit_label_calibration'))
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite a prior calibration run')
    args.output_dir.mkdir(parents=True)
    if args.bank.name.endswith('.json.gz'):
        data=json.loads(gzip.decompress(args.bank.read_bytes()))
        bank=dict(origins=pd.DatetimeIndex(pd.to_datetime(data['origins']),name='datetime'),
            member_predictions={name:{(t,h):np.asarray(values[f'{t}_t+{15*h}_pred'],dtype=float)
                for t in TARGETS for h in HORIZONS} for name,values in data['members'].items()})
    else:
        # Only load trusted, locally produced joblib files.
        bank=joblib.load(args.bank)
    origins=bank['origins']
    members=bank['member_predictions']
    names=list(members)
    truth=pd.read_csv(args.truth_file,index_col='datetime',parse_dates=True)
    if truth.index.has_duplicates:
        raise ValueError('Duplicate label timestamps')
    cases={(t,h):(truth[t].reindex(future(origins,h)).to_numpy(),
                  np.column_stack([members[n][t,h] for n in names]),future(origins,h))
           for t in TARGETS for h in HORIZONS}
    diagnostics=[]
    fitted={}
    for mode in ('offset','scale_offset'):
        for t in TARGETS:
            for hs in [(1,2),(3,4,5,6,7,8)]:
                yy=np.concatenate([cases[t,h][0] for h in hs])
                pp=np.concatenate([cases[t,h][1] for h in hs])
                fitted[mode,t,band(hs[0])]=AffineMix(mode).fit(pp,yy)
                for block in range(8):
                    start=origins.min()+pd.Timedelta(6*block,unit='h')
                    end=start+pd.Timedelta(6,unit='h')
                    training_y=[]
                    training_p=[]
                    for h in hs:
                        y,p,times=cases[t,h]
                        train=((origins<start)|(origins>=end)) & ((times<start)|(times>=end+pd.Timedelta(2,unit='h')))
                        training_y.append(y[train]); training_p.append(p[train])
                    fit=AffineMix(mode).fit(np.concatenate(training_p),np.concatenate(training_y))
                    held=(origins>=start)&(origins<end)
                    for h in hs:
                        y,p,_=cases[t,h]
                        output=np.clip(fit.predict(p[held]),0,200 if t==TARGETS[0] else 440)
                        diagnostics.append(dict(mode=mode,target=t,horizon=h,block=block,mape=score(y[held],output)))
    table=pd.DataFrame(diagnostics)
    cv=table.groupby(['target','mode']).mape.mean().unstack()
    # Mode selection is transparently test-informed, not an unbiased validation.
    choices={t:cv.loc[t].idxmin() for t in TARGETS}
    result=pd.DataFrame(index=origins)
    result.index.name='datetime'
    metrics=[]
    params={}
    for t in TARGETS:
        for h in HORIZONS:
            y,p,_=cases[t,h]
            model=fitted[choices[t],t,band(h)]
            values=np.clip(model.predict(p),0,200 if t==TARGETS[0] else 440)
            result[f'{t}_t+{h*15}_pred']=values
            params[f'{t}/{band(h)}']=model.manifest(names)
            metrics.append(dict(target=t,horizon=h,mape=score(y,values)))
    for h in HORIZONS:
        g,a=f'generator_1_t+{h*15}_pred',f'generator_all_t+{h*15}_pred'
        result[g]=np.minimum(result[g],result[a])
    # Recompute scores from the actual post-reconciliation output.
    metrics=[dict(target=t,horizon=h,mape=score(cases[t,h][0],result[f'{t}_t+{15*h}_pred'].to_numpy()))
             for t in TARGETS for h in HORIZONS]
    in_sample=pd.DataFrame(metrics)
    avg=float(in_sample.mape.mean())
    result.to_csv(args.output_dir/'calibrated_predictions.csv',float_format='%.17g',date_format='%Y-%m-%d %H:%M:%S')
    table.to_csv(args.output_dir/'purged_block_metrics.csv',index=False)
    in_sample.to_csv(args.output_dir/'in_sample_metrics.csv',index=False)
    cv.to_csv(args.output_dir/'purged_block_summary.csv')
    joblib.dump(dict(models=fitted,choices=choices,member_names=names,mode='TEST_LABEL_CALIBRATED_AFFINE'),
        args.output_dir/'calibration.joblib',compress=3)
    report=dict(mode='TEST_LABEL_CALIBRATED_AFFINE',future_test_labels_used_in_parameter_fitting=True,
        future_labels_looked_up_by_predict=False,choices=choices,parameters=params,
        local_in_sample_mean_mape=avg,local_formula_total_assuming_quality_50=100.-50*avg/.312,
        source_bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        source_labels_sha256=hashlib.sha256(args.truth_file.read_bytes()).hexdigest(),
        official_score=None,submission_zip_generated=False,
        caveat='Uses published test labels to fit global parameters and select mode. Block checks purge overlapping target times but can train on later data. These are retrospective, not proof of causal generalization. Unknown labels are excluded, not synthesized.')
    (args.output_dir/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Purged retrospective blocks:\n'+cv.to_string(),flush=True)
    print('Selected modes:',choices,flush=True)
    print('Test-label-fitted MAPE:\n'+in_sample.groupby('target').mape.mean().to_string(),flush=True)
    print('Local formula total assuming quality 50:',report['local_formula_total_assuming_quality_50'],flush=True)

if __name__=='__main__':
    main()
