"""HiGHS integer unit commitment; no LP relaxation or fabricated feasible plan."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def solve_dispatch(cfg,timestamps,prices,net_inflow,gas_caps,efficiency,initial_holder):
    from pipeline import DispatchResult, GAS_TYPES, OPT_COLUMNS
    n=len(timestamps)
    if not n:
        raise ValueError('Empty dispatch horizon')
    timestamps=pd.DatetimeIndex(timestamps)
    if hasattr(timestamps, 'as_unit'):
        timestamps = timestamps.as_unit('ns')
    if (not timestamps.is_unique or not timestamps.is_monotonic_increasing
            or timestamps.hasnans):
        raise ValueError('Dispatch timestamps must be unique, sorted and valid')
    if n>1 and not (np.diff(timestamps.asi8)==cfg.interval_minutes*60_000_000_000).all():
        raise ValueError('Dispatch timestamps do not match the configured interval')
    if tuple(cfg.unit_ratings_mw)!=(50.,50.,50.,50.,120.,120.):
        raise ValueError('This formulation requires the documented 4x50+2x120 configuration')
    dt=cfg.interval_minutes/60
    # Gas volumes in thousands of m3, power in MW, prices in currency/kWh.
    net=np.array([net_inflow[g] for g in GAS_TYPES],dtype=float)/1000
    caps=np.array([gas_caps[g] for g in GAS_TYPES],dtype=float)/1000
    eta=np.array([efficiency[g] for g in GAS_TYPES])*1000
    if net.shape!=(3,n) or not np.isfinite(net).all() or (net<0).any():
        raise ValueError('Nonnegative finite net gas supply required; deficits must be resolved upstream')
    if caps.shape!=(3,n) or eta.shape!=(3,) or not np.isfinite(caps).all() or (caps<0).any() or not np.isfinite(eta).all() or (eta<=0).any():
        raise ValueError('Invalid gas cap/efficiency')
    if np.asarray(prices).shape!=(n,) or not np.isfinite(prices).all():
        raise ValueError('Invalid tariff')
    lo=cfg.holder_capacity*cfg.holder_min_fraction/1000
    hi=cfg.holder_capacity*cfg.holder_max_fraction/1000
    initial=float(initial_holder)/1000
    if not lo<=initial<=hi:
        raise ValueError(f'Observed initial holder {initial_holder} is outside safe bounds; cannot silently clip it')
    offsets={}; total=0
    def block(name,shape):
        nonlocal total
        indices=np.arange(total,total+int(np.prod(shape))).reshape(shape)
        total+=indices.size; offsets[name]=indices
        return indices
    q=block('gas',(3,n)); v=block('stock',(n+1,)); f=block('flare',(3,n))
    p=block('power',(2,n)); u=block('online_count',(2,n)); r=block('ramp',(2,n)); start=block('start',(2,n))
    cost=np.zeros(total); lower=np.zeros(total); upper=np.full(total,np.inf); integer=np.zeros(total)
    upper[q.ravel()]=caps.ravel()
    lower[v]=lo; upper[v]=hi; lower[v[0]]=upper[v[0]]=initial
    # End with at least the initial inventory; prevents artificial horizon-end profit.
    lower[v[-1]]=initial
    upper[p[0]]=200; upper[p[1]]=240
    upper[u[0]]=4; upper[u[1]]=2; integer[u.ravel()]=1
    cost[p.ravel()]=np.tile(-np.asarray(prices)*dt*cfg.revenue_scale,2)
    cost[f.ravel()]=cfg.flare_penalty*1000*dt
    cost[r.ravel()]=cfg.ramp_penalty; cost[start.ravel()]=cfg.startup_penalty
    rr=[]; cc=[]; vv=[]; lb=[]; ub=[]
    def row(terms,low=-np.inf,high=np.inf):
        ix=len(lb)
        for c,value in terms:
            rr.append(ix);cc.append(int(c));vv.append(float(value))
        lb.append(low);ub.append(high)
    for t in range(n):
        row([(p[0,t],1),(p[1,t],1)]+[(q[g,t],-eta[g]) for g in range(3)],0,0)
        # Only blast-furnace gas has an observed holder. No cross-gas borrowing.
        row([(v[t+1],1),(v[t],-1),(q[0,t],dt),(f[0,t],dt)],dt*net[0,t],dt*net[0,t])
        for g in (1,2):
            row([(q[g,t],1),(f[g,t],1)],net[g,t],net[g,t])
        for g,rating in enumerate((50.,120.)):
            row([(p[g,t],1),(u[g,t],-rating)],high=0)
            row([(p[g,t],-1),(u[g,t],cfg.minimum_load_fraction*rating)],high=0)
            if t:
                change=[(p[g,t],1),(p[g,t-1],-1)]
                limit=cfg.ramp_fraction_per_minute*cfg.interval_minutes*(200 if g==0 else 240)
                row(change,-limit,limit)
                row(change+[(r[g,t],-1)],high=0)
                row([(c,-val) for c,val in change]+[(r[g,t],-1)],high=0)
                row([(u[g,t],1),(u[g,t-1],-1),(start[g,t],-1)],high=0)
    a=coo_matrix((vv,(rr,cc)),shape=(len(lb),total)).tocsc()
    result=milp(cost,integrality=integer,bounds=Bounds(lower,upper),
        constraints=LinearConstraint(a,np.array(lb),np.array(ub)),
        options={'time_limit':float(cfg.solver_time_limit),'mip_rel_gap':.005,'presolve':True})
    if result.x is None:
        raise RuntimeError(f'No feasible integer dispatch: {result.message}')
    z=result.x
    av=a@z
    violation=max(float(np.max(np.maximum(np.array(lb)-av,0))),
        float(np.max(np.maximum(av-np.array(ub),0))),
        float(np.max(np.maximum(lower-z,0))),float(np.max(np.maximum(z-upper,0))),
        float(np.max(np.abs(z[u.ravel()]-np.rint(z[u.ravel()])))))
    if violation>1e-4:
        raise RuntimeError(f'Unverified solver incumbent: max constraint violation {violation}')
    plan=pd.DataFrame({OPT_COLUMNS[g]:np.maximum(z[q[i]],0)*1000 for i,g in enumerate(GAS_TYPES)},index=timestamps)
    power=z[p].sum(axis=0)
    revenue=float(np.sum(power*np.asarray(prices)*dt*cfg.revenue_scale))
    diag=dict(objective=float(-result.fun),revenue=revenue,
        flare_total=float(z[f].sum()*1000*dt),emergency_total=0.,
        holder_min=float(z[v].min()*1000),holder_max=float(z[v].max()*1000),
        constraint_violation=violation,mip_gap=float(getattr(result,'mip_gap',0)),
        power_max=float(power.max()),terminal_holder=float(z[v[-1]]*1000))
    out=DispatchResult(plan,'MILP_HiGHS_optimal' if result.status==0 else 'MILP_HiGHS_verified_incumbent',z[v]*1000,power,diag)
    out.audit=pd.DataFrame({'generator_1':z[p[0]],'generator_all':power,
        'online_50mw':np.rint(z[u[0]]),'online_120mw':np.rint(z[u[1]]),
        'holder_m3':z[v[1:]]*1000, 'flare_m3h':z[f].sum(axis=0)*1000},index=timestamps)
    return out


def observed_surplus(raw,origin,horizon):
    """Forecast realized gas availability AFTER priority users, from gas burned
    and observed inventory change. This is an estimate, not a measured future
    production guarantee. Coke/converter have no observed storage in this package.
    Assumes generator gas flows are m3/h; explicit in run metadata/report.
    """
    from pipeline import GAS_TYPES
    hist=raw.loc[:origin].ffill()
    holder='blast_furnace_gas_holder_2' if ('blast_furnace_gas_holder_2' in hist and hist.blast_furnace_gas_holder_2.notna().any()) else 'blast_furnace_gas_holder_1'
    if holder not in hist or hist[holder].dropna().empty:
        raise ValueError('No observed blast-furnace holder; dispatch requires initial inventory')
    capacity=300000 if holder.endswith('_2') else 200000
    net={}; caps={}
    for gas in GAS_TYPES:
        s=hist['generator_use_'+gas+'_gas']
        if s.dropna().empty:
            raise ValueError(f'No observed generation gas: {gas}')
        available=s.copy()
        if gas=='blast_furnace':
            available=available+hist[holder].diff()/0.25
        recent=float(available.tail(16).median())
        # A priority-user deficit requires an upstream response, not invented gas.
        if not np.isfinite(recent) or recent<0:
            raise ValueError(f'{gas}: invalid or negative post-user supply estimate {recent}')
        level=recent
        net[gas]=np.full(horizon,level)
        caps[gas]=np.full(horizon,max(level,float(s.tail(96*14).quantile(.99))))
    return net,caps,float(hist[holder].iloc[-1]),capacity,holder
