"""April-only online deployment simulations with matured-label audit."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from online_model import SPECS,OnlineRidge
from compact_inputs import CompactInputTransform
from prelim import TARGETS,HORIZONS,read_data,future,score
from develop_stable import FOLDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/online_v10'))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    rows=[]
    for fold in FOLDS:
        cutoff=future([pd.Timestamp(fold)],-1)[0]
        ix=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        tx=CompactInputTransform().fit(raw.loc[:cutoff])
        x=tx.transform(raw)
        for spec in SPECS:
            tic=time.perf_counter()
            m=OnlineRidge(spec).fit(raw,x,cutoff)
            pred=m.predict(raw,x,ix)
            joblib.dump(dict(predictions=pred,spec=asdict(spec),audit=m.update_audit),args.output_dir/f'fold_{fold}_{spec.name}.joblib')
            for t in TARGETS:
                for h in HORIZONS:
                    times=future(ix,h)
                    y=raw[t].reindex(times).to_numpy()
                    y[times>ix.max()]=np.nan
                    rows.append(dict(fold=fold,target=t,horizon=h,member=spec.name,mape=score(y,pred[t,h])))
            print(f'{fold} {spec.name}: {time.perf_counter()-tic:.1f}s',flush=True)
        table=pd.DataFrame(rows)
        print(table[table.fold==fold].groupby(['target','member']).mape.mean().unstack().round(5).to_string(),flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'development_metrics.csv',index=False)

if __name__=='__main__':
    main()
