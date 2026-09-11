"""Behavioral regressions: leakage, data fidelity, tariffs, and hard dispatch constraints."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from pipeline import IndustrialDataPipeline, PriceSchedule, MILPDispatcher, DispatchConfig, GAS_TYPES
from forecasting import features, horizon_features, ResidualEnsemble, ModelConfig, SharedHorizonResidual, SharedHorizonConfig
from dispatch import observed_surplus


class Invariants(unittest.TestCase):
    def test_right_closed_resampling_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as d:
            idx=pd.to_datetime(['2025-01-01 00:00','2025-01-01 00:01','2025-01-01 00:15','2025-01-01 00:45'])
            pd.DataFrame({'datetime':idx,'generator_1':[10,999,20,30],
                'generator_all':[30,999,50,70],'unused':[np.nan]*4}).to_csv(Path(d)/'Pre_load.csv',index=False)
            loader=IndustrialDataPipeline(); clean=loader.load(d)
            self.assertEqual(clean.iloc[0].generator_1,10)
            self.assertTrue(np.isnan(loader.observations.loc['2025-01-01 00:30','generator_1']))
            self.assertIn('unused',clean)

    def test_future_mutation_cannot_change_features(self):
        idx=pd.date_range('2025-01-01',periods=800,freq='15min')
        raw=pd.DataFrame({'generator_1':50+np.sin(np.arange(800)),
            'generator_all':200+np.cos(np.arange(800))},index=idx)
        altered=raw.copy(); altered.iloc[701:]=99999
        a,b=features(raw),features(altered)
        pd.testing.assert_frame_equal(a.iloc[:701],b.iloc[:701])
        for h in (1,8,96):
            pd.testing.assert_frame_equal(horizon_features(a.iloc[690:701],raw,h),
                horizon_features(b.iloc[690:701],altered,h))

    def test_shared_horizon_model_has_purged_labels_and_no_future_dependency(self):
        idx=pd.date_range('2025-01-01',periods=900,freq='15min')
        raw=pd.DataFrame({'generator_1':50+np.sin(np.arange(900)/10),
            'generator_all':200+np.cos(np.arange(900)/9),
            'blast_furnace_1':1000+np.sin(np.arange(900))},index=idx)
        altered=raw.copy(); altered.iloc[721:]=99999
        cfg=SharedHorizonConfig(trees=3,threads=1,train_days=4,origin_stride=4,max_features=30)
        x,z=features(raw),features(altered)
        a=SharedHorizonResidual(cfg).fit(raw,x,idx[720],range(1,9))
        b=SharedHorizonResidual(cfg).fit(altered,z,idx[720],range(1,9))
        self.assertTrue(all(pd.Timestamp(row['last_label'])<=idx[720] for row in a.training_audit))
        for target in ('generator_1','generator_all'):
            np.testing.assert_allclose(a.predict(raw,x,idx[720:721],8,target),
                b.predict(altered,z,idx[720:721],8,target),rtol=0,atol=1e-9)

    def test_recency_weighted_residual_model_fits(self):
        idx=pd.date_range('2025-01-01',periods=800,freq='15min')
        raw=pd.DataFrame({'generator_1':50+np.sin(np.arange(800)/10),
            'generator_all':200+np.cos(np.arange(800)/9)},index=idx)
        model=ResidualEnsemble(ModelConfig(trees=3,threads=1,train_days=4,
            recency_half_life_days=2)).fit(raw,features(raw),idx[700],[1,8])
        self.assertEqual(len(model.training_audit),4)

    def test_target_specific_recency_configuration_fits(self):
        idx=pd.date_range('2025-01-01',periods=800,freq='15min')
        raw=pd.DataFrame({'generator_1':50+np.sin(np.arange(800)/10),
            'generator_all':200+np.cos(np.arange(800)/9)},index=idx)
        cfg=ModelConfig(trees=3,threads=1,train_days=4,
            recency_half_life_by_target={'generator_1':2.,'generator_all':None})
        model=ResidualEnsemble(cfg).fit(raw,features(raw),idx[700],[1])
        self.assertEqual(len(model.training_audit),2)

    def test_tariffs_preserve_discontinuity(self):
        with tempfile.TemporaryDirectory() as d:
            f=pd.DataFrame({'time':[str(i) for i in range(48)],'1月':[.2]*24+[1.1]*24,'2月':[.3]*48})
            p=Path(d)/'price.xlsx';f.to_excel(p,index=False)
            schedule=PriceSchedule.from_excel(p)
            actual=schedule.prices(pd.to_datetime(['2025-01-01 11:45','2025-01-01 12:00']))
            np.testing.assert_array_equal(actual,[.2,1.1])

    def test_fitted_model_cannot_learn_future_and_survives_reload(self):
        idx=pd.date_range('2025-01-01',periods=800,freq='15min')
        raw=pd.DataFrame({'generator_1':50+np.sin(np.arange(800)/10),
            'generator_all':200+np.cos(np.arange(800)/9)},index=idx)
        raw.iloc[300,0]=np.nan
        changed=raw.copy(); changed.iloc[651:]=90000
        cfg=ModelConfig(trees=3,threads=1)
        x,z=features(raw),features(changed)
        a=ResidualEnsemble(cfg).fit(raw,x,idx[650],[1,96])
        b=ResidualEnsemble(cfg).fit(changed,z,idx[650],[1,96])
        self.assertTrue(all(pd.Timestamp(row['last_label'])<=idx[650] for row in a.training_audit))
        for h in (1,96):
            for t in ('generator_1','generator_all'):
                np.testing.assert_allclose(a.predict_members(raw,x,idx[650:651],h,t),
                    b.predict_members(changed,z,idx[650:651],h,t),rtol=0,atol=1e-9)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.joblib'; a.save(p); loaded=joblib.load(p)
            np.testing.assert_array_equal(a.predict_members(raw,x,idx[650:651],96,'generator_1'),
                loaded.predict_members(raw,x,idx[650:651],96,'generator_1'))

    def test_tariff_missing_file_or_month_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                PriceSchedule.from_excel(Path(d)/'missing.xlsx')
        with self.assertRaises(ValueError):
            PriceSchedule({1:np.ones(96)}).prices(pd.to_datetime(['2025-02-01']))

    def test_96_step_integer_dispatch_mass_balance(self):
        n=96; idx=pd.date_range('2025-01-01',periods=n,freq='15min')
        net={g:np.full(n,v) for g,v in zip(GAS_TYPES,(700000,10000,20000))}
        caps={g:v*1.8 for g,v in net.items()}
        eta=dict(zip(GAS_TYPES,(.0003,.0015,.0007)))
        cfg=DispatchConfig(solver_time_limit=12)
        result=MILPDispatcher(cfg).optimize(idx,np.where(idx.hour<12,.2,1.1),net,caps,eta,100000)
        self.assertTrue(result.status.startswith('MILP_HiGHS'))
        self.assertLessEqual(result.diagnostics['constraint_violation'],1e-4)
        self.assertGreaterEqual(result.holder_path.min(),30000-1e-4)
        self.assertLessEqual(result.holder_path.max(),180000+1e-4)
        self.assertGreaterEqual(result.holder_path[-1],100000-1e-4)
        self.assertLessEqual(result.power_path.max(),440+1e-4)
        for g in ('coke','converter'):
            self.assertTrue((result.gas_plan['opt_generator_use_'+g+'_gas']<=net[g]+1e-4).all())
        a=result.audit
        for count,power,rating in [(a.online_50mw,a.generator_1,50),
                (a.online_120mw,a.generator_all-a.generator_1,120)]:
            self.assertTrue((power>=.6*rating*count-1e-4).all())
            self.assertTrue((power<=rating*count+1e-4).all())

    def test_invalid_inventory_not_silently_clipped(self):
        n=8; idx=pd.date_range('2025-01-01',periods=n,freq='15min')
        net={g:np.ones(n) for g in GAS_TYPES}; eta={g:.001 for g in GAS_TYPES}
        with self.assertRaises(ValueError):
            MILPDispatcher().optimize(idx,np.ones(n),net,net,eta,1.)

    def test_priority_user_deficit_not_hidden(self):
        idx=pd.date_range('2025-01-01',periods=20,freq='15min')
        raw=pd.DataFrame({'blast_furnace_gas_holder_2':150000-np.arange(20)*1000,
            **{'generator_use_'+g+'_gas':np.ones(20)*100 for g in GAS_TYPES}},index=idx)
        with self.assertRaises(ValueError):
            observed_surplus(raw,idx[-1],96)


if __name__=='__main__':
    unittest.main()
