import time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, h4d
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
D = {c: h4d.load_cube(c) for c in ['Madrid', 'Amsterdam']}
y = {c: h4d.age_class(D[c]['wmy'], c) for c in D}
F = {c: pd.read_parquet(f'data/cache/feat_{c}.parquet') for c in D}
R = {c: pd.read_parquet(f'data/cache/raw_{c}.parquet') for c in D}
def cityz(df): return (df - df.median()) / (df.quantile(.75) - df.quantile(.25) + 1e-9)
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, class_weight='balanced', n_jobs=4, verbose=-1, random_state=42)
ctx = [c for c in F['Madrid'].columns if c.startswith('ctx')]
variants = {
  'raw absolute':            (R['Madrid'], R['Amsterdam']),
  'raw + per-city z':        (cityz(R['Madrid']), cityz(R['Amsterdam'])),
  'anomaly (no ctx)':        (F['Madrid'].drop(columns=ctx), F['Amsterdam'].drop(columns=ctx)),
  'anomaly + ctx':           (F['Madrid'], F['Amsterdam']),
  'anomaly + ctx + city z':  (cityz(F['Madrid']), cityz(F['Amsterdam'])),
}
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for name, (Xm, Xa) in variants.items():
    t = time.time(); f1s = []
    for tr, va in list(skf.split(Xm, y['Madrid']))[:2]:
        m = lgb.LGBMClassifier(**P).fit(Xm.iloc[tr], y['Madrid'][tr])
        f1s.append(f1_score(y['Madrid'][va], m.predict(Xm.iloc[va]), average='macro'))
    m = lgb.LGBMClassifier(**P).fit(Xm, y['Madrid'])
    pa = m.predict(Xa)
    print(f'{name:26s} Madrid CV(2 folds) {np.mean(f1s):.4f} | Ams zero-shot {f1_score(y["Amsterdam"], pa, average="macro"):.4f}'
          f'  pred dist {np.bincount(pa, minlength=5)[1:]}  {time.time()-t:.0f}s', flush=True)
