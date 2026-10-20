"""Compare few-shot transfer strategies on identical support draws."""
import sys, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, h4d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

D = {c: h4d.load_cube(c) for c in ['Madrid', 'Amsterdam']}
y = {c: h4d.age_class(D[c]['wmy'], c) for c in D}
F = {c: pd.read_parquet(f'data/cache/feat_{c}.parquet') for c in D}
R = {c: pd.read_parquet(f'data/cache/raw_{c}.parquet') for c in D}
cityz = lambda df: (df - df.median()) / (df.quantile(.75) - df.quantile(.25) + 1e-9)
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'cityz'
def pick(c):
    if VARIANT == 'cityz': return cityz(F[c])
    if VARIANT == 'anom':  return F[c]
    if VARIANT == 'raw':   return R[c]
    if VARIANT == 'both':  return pd.concat([R[c].add_prefix('abs_'), F[c]], axis=1)
X = {c: pick(c) for c in D}
print('VARIANT', VARIANT, X['Madrid'].shape, flush=True)
Xm, Xa, ym, ya = X['Madrid'], X['Amsterdam'], y['Madrid'], y['Amsterdam']

P = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, class_weight='balanced', n_jobs=4, verbose=-1, random_state=42)
t = time.time()
clf = lgb.LGBMClassifier(**P).fit(Xm, ym)
reg = lgb.LGBMRegressor(**{**P, 'class_weight': None}).fit(Xm, D['Madrid']['wmy'])
print(f'Madrid models trained {time.time()-t:.0f}s', flush=True)
L = np.log(clf.predict_proba(Xa) + 1e-6)              # Madrid class log-probs on Amsterdam
raw_score = clf.predict_proba(Xa, raw_score=True)
yr = reg.predict(Xa)[:, None]                          # Madrid "construction year" score
Z = StandardScaler().fit_transform(np.hstack([L, yr]))
print('zero-shot', round(f1_score(ya, clf.predict(Xa), average='macro'), 4), flush=True)

# Baseline-style prototype on raw absolute features with a Madrid-fit scaler
sc = StandardScaler().fit(R['Madrid']); Ra = sc.transform(R['Amsterdam'])

def proto(Xs, ys, Xq):
    cls = np.unique(ys); P_ = np.stack([Xs[ys == c].mean(0) for c in cls])
    return cls[((Xq[:, None] - P_[None]) ** 2).sum(2).argmin(1)]

def lr_on(Zs, ys, Zq, C):
    return LogisticRegression(C=C, max_iter=2000, class_weight='balanced').fit(Zs, ys).predict(Zq)

def finetune(sup, n):
    """Continue boosting from the Madrid model's scores on the Amsterdam support set."""
    ds = lgb.Dataset(Xa.values[sup], ya[sup] - 1, init_score=raw_score[sup], free_raw_data=False)
    p = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=7,
             min_data_in_leaf=max(3, n // 5), feature_fraction=0.5, lambda_l2=5, verbose=-1,
             num_threads=4, seed=42)
    b = lgb.train(p, ds, num_boost_round=int(np.clip(n, 10, 150)))
    s = b.predict(Xa.values, raw_score=True) + raw_score
    return s.argmax(1) + 1

SHOTS, TRIALS = [5, 25, 50, 100, 200], 10
rng = np.random.default_rng(42)
methods = ['proto_raw (baseline)', 'proto_engineered', 'timeline_LR[year only]', 'LR[madrid logits+year]', 'LR[logits+year]+proto',
           'finetune_gbm', 'LR[feats]']
res = {m: {n: [] for n in SHOTS} for m in methods}
for n in SHOTS:
    for tr in range(TRIALS):
        sup = np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in range(1, 5)])
        q = np.ones(len(ya), bool); q[sup] = False
        ys, yq = ya[sup], ya[q]
        f = lambda p: f1_score(yq, p, average='macro')
        res['proto_raw (baseline)'][n].append(f(proto(Ra[sup], ys, Ra[q])))
        res['proto_engineered'][n].append(f(proto(Xa.values[sup], ys, Xa.values[q])))
        C = 0.05 if n <= 5 else 0.3 if n <= 50 else 1.0
        Zy = Z[:, -1:]
        res['timeline_LR[year only]'][n].append(f(lr_on(Zy[sup], ys, Zy[q], 1.0)))
        res['LR[madrid logits+year]'][n].append(f(lr_on(Z[sup], ys, Z[q], C)))
        # Ensemble: average LR probability with prototype softmax
        lr = LogisticRegression(C=C, max_iter=2000, class_weight='balanced').fit(Z[sup], ys)
        pl = lr.predict_proba(Z[q])
        Xe = Xa.values; Pr = np.stack([Xe[sup][ys == c].mean(0) for c in range(1, 5)])
        dist = np.sqrt(((Xe[q][:, None] - Pr[None]) ** 2).sum(2)); dist = -(dist - dist.min(1, keepdims=True))
        pp = np.exp(dist / dist.std()); pp /= pp.sum(1, keepdims=True)
        res['LR[logits+year]+proto'][n].append(f((0.5 * pl + 0.5 * pp).argmax(1) + 1))
        res['finetune_gbm'][n].append(f(finetune(sup, n)[q]))
        Xs_ = np.hstack([Z, Xa.values])
        res['LR[feats]'][n].append(f(lr_on(Xs_[sup], ys, Xs_[q], 0.01 if n <= 25 else 0.05)))
    print(f'n={n:3d} ' + ' | '.join(f'{m}: {np.mean(res[m][n]):.3f}±{np.std(res[m][n]):.3f}' for m in methods), flush=True)

summary = pd.DataFrame({m: [np.mean(res[m][n]) for n in SHOTS] for m in methods}, index=SHOTS).T
summary['avg'] = summary.mean(1)
print(summary.round(4).to_string())
