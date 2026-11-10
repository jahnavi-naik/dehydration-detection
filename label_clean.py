#!/usr/bin/env python3
# Join fastingTimes.csv labels to features_export/features.csv and drop windows from
# recordings with a broken sampling clock. Run from the folder holding features_export/.
# Output: labeled_clean.csv. Then: python analysis.py features_export --labeled-features labeled_clean.csv --drop hr_mean hr_std ibi_sdnn_ms ibi_rmssd_ms
# (S6 is dropped automatically by analysis.py: it has no hydrated baseline.)
import pandas as pd, numpy as np, glob, re, sys
E='features_export'
f=pd.read_csv(f'{E}/features.csv'); m=pd.read_csv(f'{E}/manifest.csv')
print('raw windows',len(f))
# fasting start times
ft=pd.concat([pd.read_csv(p,dtype=str).assign(subject=p.split('/')[-1].split('_')[0]) for p in glob.glob(f'{E}/fasting_times/*.csv')])
ft['rec']=ft['RecordingTime'].str.strip(); ft['fast_start']=pd.to_datetime(ft['FastingTime'].str.strip(),format='%Y%m%d%H%M%S')
f['rec']=f['file'].str[:14]
f=f.merge(ft[['subject','rec','fast_start']],on=['subject','rec'],how='left')
fa=f[f.condition=='fasting']
print('fasting windows w/o fast_start:',fa.fast_start.isna().sum(), fa[fa.fast_start.isna()].file.unique())
# cleaning
exp=60*51.2
bad_fs=m[(m.fs_hz-51.2).abs()>0.5].file
c0=len(f)
f['clean_reason']=''
f.loc[f.file.isin(bad_fs),'clean_reason']='bad sampling clock'
f.loc[(f.clean_reason=='')&((f.n_samples<0.8*exp)|(f.n_samples>1.05*exp)),'clean_reason']='timestamp gap/jump'
f['window_start']=pd.to_datetime(f.window_start)
f['hours_since_drink']=np.where(f.condition=='fasting',(f.window_start-f.fast_start).dt.total_seconds()/3600,1.0)
f.loc[(f.clean_reason=='')&(f.hours_since_drink<0),'clean_reason']='label before fast start'
print(f.clean_reason.value_counts())
print(f[f.clean_reason!=''].groupby(['subject','file']).size().sort_values().tail(12))
g=f[f.clean_reason==''].copy()
g.to_csv('labeled_clean.csv',index=False)
print('clean windows',len(g), 'minutes',len(g))
print(g.groupby(['subject','condition']).size().unstack(fill_value=0))
print(g[g.condition=='fasting'].groupby('subject').hours_since_drink.describe().round(1))
for t in [6,8,10,12]:
    print(t, g.assign(y=g.hours_since_drink>=t).groupby('subject').y.sum().to_dict())
