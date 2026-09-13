"""Periodic causal refitting; each update uses labels strictly before its origin.

This is an explicit rolling-observation deployment protocol, not frozen-model
inference. Hyperparameters and preprocessing are frozen in April; fitted Ridge
coefficients may be updated when new observations have actually arrived.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
from prelim import Spec,ShortModel,TARGETS,HORIZONS,future

SPECS=[Spec('online7_6h','online',days=7,alpha=300.),
       Spec('online21_6h','online',days=21,alpha=100.),
       Spec('online60_6h','online',days=60,alpha=300.,half_life=14.),
       Spec('online21_2h','online',days=21,alpha=100.)]


class OnlineRidge:
    def __init__(self,spec,threads=1):
        self.spec,self.threads=spec,threads
        self.refresh_steps=8 if spec.name.endswith('_2h') else 24

    def fit(self,raw,x,cutoff):
        self.cutoff=pd.Timestamp(cutoff)
        self.history_raw=raw.loc[:cutoff,list(TARGETS)].copy()
        self.history_x=x.loc[:cutoff].copy()
        self.inner_spec=replace(self.spec,kind='ridge')
        self.initial=ShortModel(self.inner_spec,self.threads).fit(self.history_raw,self.history_x,cutoff)
        self.audit=list(self.initial.audit)
        return self

    def predict(self,raw,x,origins):
        origins=pd.DatetimeIndex(origins)
        first=future([self.cutoff],1)[0]
        if len(origins)==0 or origins.min()<first:
            raise ValueError('Online origins must follow the initial training cutoff')
        if not origins.is_monotonic_increasing or origins.has_duplicates:
            raise ValueError('Online origins must be sorted and unique')
        merged_x=pd.concat([self.history_x,x.loc[(x.index>self.cutoff)&(x.index<=origins.max())]])
        merged_raw=pd.concat([self.history_raw,raw.loc[(raw.index>self.cutoff)&(raw.index<=origins.max()),list(TARGETS)]])
        if merged_x.index.has_duplicates or merged_raw.index.has_duplicates:
            raise ValueError('Duplicate online observations')
        steps=np.asarray((origins-first).total_seconds()/900)
        if not np.equal(steps,np.floor(steps)).all():
            raise ValueError('Online origins must align to the 15-minute grid')
        block=steps.astype(int)//self.refresh_steps
        out={(t,h):np.empty(len(origins)) for t in TARGETS for h in HORIZONS}
        self.update_audit=[]
        for b in np.unique(block):
            mask=block==b
            query=origins[mask]
            update_origin=future([first],int(b)*self.refresh_steps)[0]
            cutoff=future([update_origin],-1)[0]
            if b==0:
                model=self.initial
            else:
                needed=pd.date_range(first,cutoff,freq='15min')
                if not needed.isin(merged_x.index).all() or not needed.isin(merged_raw.index).all():
                    raise ValueError('Missing arrived observations needed for online replay')
                # Truncate before fitting so even schema/weight fitting cannot
                # see data after this scheduled update.
                model=ShortModel(self.inner_spec,self.threads).fit(merged_raw.loc[:cutoff],merged_x.loc[:cutoff],cutoff)
            self.update_audit.extend([{**a,'update_origin':str(update_origin)} for a in model.audit])
            p=model.predict(merged_raw,merged_x,query)
            for key in out:
                out[key][mask]=p[key]
        return out
