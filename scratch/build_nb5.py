"""Generates '5-Our_Model (DSRH2026).ipynb'. Edit the cell texts here and re-run to rebuild."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip('\n')))

# ════════════════════════════════════════════════════════════════════════════
md(r"""
# 🏗️ Building Age Classification from Satellite Imagery

# Notebook 5 — Our Model: Further EDA, Transfer-Oriented Features & Cross-City Transfer

---

Notebooks 1–4 are the organisers' starter material. This notebook is **our own solution**. It has three parts:

| Part | What we do | Why |
|---|---|---|
| **A. Further EDA** | Find out *why* the Madrid model collapses on Amsterdam | We can only fix the domain gap once we understand it |
| **B. Feature engineering** | City-relative time series, change-point (construction-event) features, neighbourhood context | Give the model signals that mean the same thing in any city |
| **C. Two-stage transfer model** | Stage 1: Madrid models (age-class classifier + **construction-year regressor**). Stage 2: a small Amsterdam calibrator learned from *n* labelled pixels per class | The few-shot step genuinely re-uses what Madrid taught us |

### Baseline to beat (Notebook 4)

| Setting | Macro-F1 |
|---|---|
| Madrid, 5-fold × 5-repeat CV | 0.618 ± 0.004 |
| Amsterdam zero-shot | 0.343 |
| Amsterdam prototype, 5 / 25 / 50 / 100 / 200 per class | 0.544 / 0.599 / 0.608 / 0.613 / 0.615 |

The baseline's few-shot step only measures distances to Amsterdam class means, so **it never uses the Madrid model**. Our Stage 2 is built on top of the Madrid model's outputs.
""")

md(r"""
## 5.1 Setup

Everything is controlled from the config cell below. `SEED` fixes every random draw, so the notebook is reproducible end-to-end.

**Memory note.** The raw parquet files expand to several GB in pandas. We load only the numeric columns we need and pack each city into a dense NumPy array (the "cube") of shape `(pixels, years, bands)`. This takes about 80 MB for Madrid. The cube is cached to `data/cache/`, so re-runs are fast.
""")

code(r"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

warnings.filterwarnings('ignore', category=RuntimeWarning)   # all-NaN medians are expected
warnings.filterwarnings('ignore', message='X does not have valid feature names')
plt.rcParams['figure.dpi'] = 100

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
SHOTS        = [5, 25, 50, 100, 200]      # labelled Amsterdam pixels per class
N_TRIALS     = 10                         # random support draws per shot size
N_FOLDS      = 5                          # Madrid cross-validation folds
N_REPEATS    = 1                          # raise to 5 to match Notebook 4 exactly (slower)
N_JOBS       = 4                          # CPU threads for LightGBM

# ── Paths: works from the repo root or a subfolder ───────────────────────────
DATA_DIR  = next(p for p in [Path('data'), Path('../data')]
                 if (p / 'madrid_train.parquet').exists())
CACHE_DIR = DATA_DIR / 'cache';  CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path('models');      MODEL_DIR.mkdir(exist_ok=True)
FILES     = {'Madrid': 'madrid_train.parquet', 'Amsterdam': 'amsterdam_data.parquet'}

# ── Domain constants ──────────────────────────────────────────────────────────
BANDS        = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
INDICES      = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']
CHANNELS     = BANDS + INDICES
YEARS        = np.arange(1984, 2026)      # 42 annual time steps
BLUE_MAX     = 15_000                     # haze threshold (Notebook 2)
CLASS_EVENTS = {'Amsterdam': 1945, 'Madrid': 1960}
CLASS_COLORS = {1: '#4393c3', 2: '#2ca25f', 3: '#fd8d3c', 4: '#de2d26'}
CITY_COLORS  = {'Madrid': '#d95f02', 'Amsterdam': '#1b9e77'}

print('Data folder:', DATA_DIR.resolve())
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
## 5.2 Loading the data into per-city cubes

For every pixel and year we combine the up-to-3 summer observations into **one value per band**:

1. We keep an observation only if it is QA-valid, has no zero or saturated band, and passes the haze filter (Blue ≤ 15,000).
2. We take the **median** of the surviving observations. Notebooks 2–4 took only the *first* valid observation. The median uses all the data and is robust to one bad scene.

Years with no valid observation stay `NaN` and are interpolated later (section 5.6).
""")

code(r"""
def age_class(year, city):
    # Bin weighted_mean_year into classes 1-4 using the city-specific class-1/2 boundary.
    bins = [-np.inf, CLASS_EVENTS[city], 1984, 2004, np.inf]
    return pd.cut(year, bins=bins, labels=[1, 2, 3, 4]).astype(int)


def load_cube(city, use_cache=True):
    # Returns a dict:
    #   cube[i, t, b]  median valid reflectance of pixel i, year YEARS[t], band b (NaN if none)
    #   n_obs[i, t]    number of valid observations behind that median (0-3)
    #   doy[i, t]      mean day-of-year of those observations
    #   px, py, x, y, coverage, wmy   per-pixel constants (wmy = weighted_mean_year)
    cache_path = CACHE_DIR / f'cube_{city}.npz'
    if use_cache and cache_path.exists():
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    # Read only the numeric columns we need (string columns are the memory hogs)
    cols = ['year', 'px_key', 'py_key', 'pixel_x', 'pixel_y', 'coverage', 'weighted_mean_year']
    for s in (1, 2, 3):
        cols += [f'{b}_{s}' for b in BANDS] + [f'qa_valid_{s}', f'doy_{s}']
    t = pq.read_table(DATA_DIR / FILES[city], columns=cols)

    # Map each row to (pixel index, year index)
    px, py = t['px_key'].to_numpy(), t['py_key'].to_numpy()
    keys = px.astype(np.int64) * 1_000_000 + py
    _, pix_idx = np.unique(keys, return_inverse=True)
    yr_idx = t['year'].to_numpy() - YEARS[0]
    n_pix = pix_idx.max() + 1

    # obs: (rows, 3 slots, 6 bands)
    obs = np.stack([np.stack([t[f'{b}_{s}'].to_numpy(zero_copy_only=False).astype(np.float32)
                              for b in BANDS], axis=1) for s in (1, 2, 3)], axis=1)
    valid = np.stack([t[f'qa_valid_{s}'].to_numpy(zero_copy_only=False) for s in (1, 2, 3)], axis=1)
    valid = np.nan_to_num(valid.astype(float), nan=0).astype(bool)
    valid &= np.isfinite(obs).all(axis=2)
    valid &= (obs > 0).all(axis=2) & (obs < 65535).all(axis=2)   # corrupt zero / saturated pixels
    valid &= obs[:, :, 0] <= BLUE_MAX                             # haze (Notebook 2)
    obs[~valid] = np.nan
    doy = np.stack([t[f'doy_{s}'].to_numpy(zero_copy_only=False).astype(np.float32)
                    for s in (1, 2, 3)], axis=1)
    doy[~valid] = np.nan

    cube = np.full((n_pix, len(YEARS), len(BANDS)), np.nan, dtype=np.float32)
    cube[pix_idx, yr_idx] = np.nanmedian(obs, axis=1)
    n_obs = np.zeros((n_pix, len(YEARS)), dtype=np.int8)
    n_obs[pix_idx, yr_idx] = valid.sum(axis=1)
    doy_c = np.full((n_pix, len(YEARS)), np.nan, dtype=np.float32)
    doy_c[pix_idx, yr_idx] = np.nanmean(doy, axis=1)

    first = np.unique(pix_idx, return_index=True)[1]          # first row of each pixel
    out = dict(cube=cube, n_obs=n_obs, doy=doy_c,
               px=px[first].astype(np.int32), py=py[first].astype(np.int32),
               x=t['pixel_x'].to_numpy()[first].astype(np.float64),
               y=t['pixel_y'].to_numpy()[first].astype(np.float64),
               coverage=t['coverage'].to_numpy()[first].astype(np.float32),
               wmy=t['weighted_mean_year'].to_numpy()[first].astype(np.float32))
    np.savez_compressed(cache_path, **out)
    return out


t0 = time.time()
DATA = {city: load_cube(city) for city in ['Madrid', 'Amsterdam']}
Y    = {city: age_class(DATA[city]['wmy'], city) for city in DATA}
print(f'Loaded in {time.time() - t0:.0f}s')
for city, d in DATA.items():
    print(f"{city:10s} cube {d['cube'].shape}  ({d['cube'].nbytes / 1e6:.0f} MB)  "
          f"class counts {np.bincount(Y[city])[1:].tolist()}  "
          f"construction years {d['wmy'].min():.0f}-{d['wmy'].max():.0f}")
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part A — Further EDA: understanding the domain gap

Notebook 4 showed that the Madrid model collapses on Amsterdam and predicts class 2 for almost everything. Before trying to fix it, we ask **what is actually different between the two cities**.
""")

md(r"""
## 5.3 Amsterdam is darker in every band, but the age ordering is the same

For each pixel we average its reflectance over all years. Then we average per class and per city.
""")

code(r"""
fig, axes = plt.subplots(1, len(BANDS), figsize=(20, 4), sharey=False)
for ax, (b, band) in zip(axes, enumerate(BANDS)):
    for city, d in DATA.items():
        pix_mean = np.nanmean(d['cube'][:, :, b], axis=1)
        means = [np.nanmean(pix_mean[Y[city] == k]) for k in range(1, 5)]
        ax.plot(range(1, 5), means, 'o-', color=CITY_COLORS[city], label=city, linewidth=2)
    ax.set_title(band, fontweight='bold'); ax.set_xticks(range(1, 5)); ax.set_xlabel('Age class')
    ax.grid(alpha=.3)
axes[0].set_ylabel('Mean reflectance (OLI scale)'); axes[0].legend()
fig.suptitle('Class-mean reflectance per city: different level, same ordering', fontweight='bold')
plt.tight_layout(); plt.show()
""")

md(r"""
**Observations**

* In **every band**, all Amsterdam classes are darker than *all* Madrid classes. In NIR, Amsterdam's newest class sits below Madrid's oldest class.
* Within each city the ordering is the same: **class 1 < 2 < 3 < 4** (newer buildings are brighter).

**Why the Madrid model collapses:** its decision thresholds are absolute reflectance values. Amsterdam pixels land in one corner of Madrid's feature space, and nearly all of them get the same class. The information that *does* transfer is **where a pixel ranks within its own city**, not its absolute brightness.
""")

md(r"""
## 5.4 Year-to-year city-wide noise and acquisition differences

If the whole city brightens or darkens in a given year, that is atmosphere, sensor or season, not buildings. Below is the city-wide median reflectance per year.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for city, d in DATA.items():
    axes[0].plot(YEARS, np.nanmedian(d['cube'][:, :, 3], axis=0), 'o-', ms=3,
                 color=CITY_COLORS[city], label=city)
    axes[1].hist(d['doy'][np.isfinite(d['doy'])], bins=60, alpha=.6, density=True,
                 color=CITY_COLORS[city], label=city)
    axes[2].hist(d['wmy'], bins=np.arange(1850, 2025, 5), alpha=.6, density=True,
                 color=CITY_COLORS[city], label=city)
axes[0].set_title('City-wide median NIR per year', fontweight='bold'); axes[0].set_xlabel('Year')
axes[1].set_title('Acquisition day-of-year', fontweight='bold'); axes[1].set_xlabel('Day of year')
axes[2].set_title('Construction year (weighted_mean_year)', fontweight='bold')
for b in (1945, 1960, 1984, 2004):
    axes[2].axvline(b, color='k', ls=':', lw=1)
for ax in axes: ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()

for city, d in DATA.items():
    print(f"{city:10s} mean valid obs/year {d['n_obs'].mean():.2f}   "
          f"mean DOY {np.nanmean(d['doy']):.0f}   latest building {d['wmy'].max():.0f}")
""")

md(r"""
**Observations**

* The city median jumps around from year to year by up to ~10% (atmosphere, sensor generation, scene date). These shifts are **shared by every pixel in the city**, so they carry no information about building age. Removing them should reduce noise.
* Amsterdam scenes are taken about 3 weeks earlier in summer on average (DOY ≈ 178 vs 202). It also has fewer clean observations per year (cloudier).
* The **newest Amsterdam building is from 2016** (Madrid: 2021), so Amsterdam's class 4 really means 2004–2016.

➡️ **Design decision:** express every pixel-year **relative to its city in that same year** (section 5.6).
""")

md(r"""
## 5.5 Building age is spatially clustered

Neighbourhoods tend to be built in the same era. If this holds, the spectral signature of a pixel's **neighbours** is extra evidence about its own age, and this is true in any city.
""")

code(r"""
def to_grid(d, values, fill=np.nan):
    gx, gy = d['px'] - d['px'].min(), d['py'] - d['py'].min()
    img = np.full((gy.max() + 1, gx.max() + 1), fill, dtype=np.float32)
    img[gy, gx] = values
    return img[::-1]          # flip so north is up

from matplotlib.colors import ListedColormap
cmap = ListedColormap([CLASS_COLORS[k] for k in range(1, 5)]); cmap.set_bad('#eeeeee')
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, (city, d) in zip(axes, DATA.items()):
    ax.imshow(to_grid(d, Y[city].astype(float)), cmap=cmap, vmin=.5, vmax=4.5, interpolation='nearest')
    ax.set_title(f'{city}: age class of every building pixel', fontweight='bold'); ax.axis('off')
plt.tight_layout(); plt.show()

# Quantify it: how often does a pixel share its class with its 8 neighbours?
for city, d in DATA.items():
    g = to_grid(d, Y[city].astype(float))
    same = tot = 0
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        a = g[1:-1, 1:-1]; b = np.roll(np.roll(g, dy, 0), dx, 1)[1:-1, 1:-1]
        ok = np.isfinite(a) & np.isfinite(b)
        same += (a[ok] == b[ok]).sum(); tot += ok.sum()
    chance = ((np.bincount(Y[city])[1:] / len(Y[city])) ** 2).sum()
    print(f'{city:10s}: neighbour shares the class {same / tot:.1%} of the time (chance: {chance:.1%})')
""")

md(r"""
**Observation:** neighbouring pixels share an age class far more often than chance in both cities. **Spatial context** is a strong, city-independent signal, so we add neighbourhood features (section 5.7).
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part B — Transfer-oriented feature engineering
""")

md(r"""
## 5.6 Time-series preparation

1. **Gap filling:** linear interpolation between observed years. Before the first and after the last observation we use the nearest value.
2. **Spectral indices:** NDVI, NDBI, UI, MNDWI and BSI (see Notebook 3), computed for every year.
3. **City-year anomaly:** for each year and channel, we subtract the city-wide median and divide by the city-wide interquartile range:

$$a_{i,t,c} = \frac{x_{i,t,c} - \mathrm{median}_j\,x_{j,t,c}}{\mathrm{IQR}_j\,x_{j,t,c}}$$

The result says *"how much brighter or darker than a typical building in this city, this year"*. That removes city-level offsets (climate, materials) and year-level noise (atmosphere, sensor) in one step.
""")

code(r"""
def fill_gaps(cube):
    # Vectorised linear interpolation along the year axis; edges take the nearest valid value.
    c = cube.copy()
    n_pix, n_yr, _ = c.shape
    t = np.arange(n_yr)
    valid = np.isfinite(c[:, :, 0])
    prev = np.maximum.accumulate(np.where(valid, t, -1), axis=1)                    # last valid <= t
    nxt = np.minimum.accumulate(np.where(valid, t, n_yr)[:, ::-1], axis=1)[:, ::-1]  # next valid >= t
    prev_c = np.where(prev < 0, nxt, prev)
    nxt_c = np.where(nxt >= n_yr, prev_c, nxt)
    rows = np.arange(n_pix)[:, None]
    lo, hi = c[rows, prev_c], c[rows, nxt_c]
    w = np.where(nxt_c > prev_c, (t - prev_c) / np.maximum(nxt_c - prev_c, 1), 0.0)[:, :, None]
    return np.where(valid[:, :, None], c, lo + w * (hi - lo)).astype(np.float32)


def add_indices(cube):
    # Append the 5 normalised-difference spectral indices as extra channels.
    B, G, R, N, S1, S2 = [cube[:, :, i] for i in range(6)]
    nd = lambda a, b: (a - b) / (a + b + 1e-6)
    idx = np.stack([nd(N, R), nd(S1, N), nd(S2, N), nd(G, S1), nd(S1 + R, N + B)], axis=2)
    return np.concatenate([cube, idx.astype(np.float32)], axis=2)


def city_year_anomaly(series):
    # Robust z-score of every value against all pixels of the same city in the same year.
    med = np.nanmedian(series, axis=0, keepdims=True)
    q75, q25 = np.nanpercentile(series, [75, 25], axis=0, keepdims=True)
    return ((series - med) / (q75 - q25 + 1e-6)).astype(np.float32)


SERIES = {city: add_indices(fill_gaps(d['cube'])) for city, d in DATA.items()}
ANOM   = {city: city_year_anomaly(s) for city, s in SERIES.items()}

# Visual check: class-mean NIR before and after the anomaly transform
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
for ax, src, title in [(axes[0], SERIES, 'Absolute NIR'), (axes[1], ANOM, 'City-year anomaly NIR')]:
    for city, ls in [('Madrid', '-'), ('Amsterdam', '--')]:
        for k in range(1, 5):
            ax.plot(YEARS, src[city][Y[city] == k, :, 3].mean(0), ls, color=CLASS_COLORS[k],
                    label=f'{city} C{k}' if city == 'Madrid' or k == 1 else None)
    ax.set_title(f'{title}  (solid = Madrid, dashed = Amsterdam)', fontweight='bold'); ax.grid(alpha=.3)
axes[0].legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
**Left:** in absolute NIR, the year-to-year noise (shared by all classes) is larger than the gaps between classes. The Amsterdam curves sit below the Madrid ones.

**Right:** after the anomaly transform, the shared noise is gone and the two cities' curves line up:

* The class ordering (1 lowest → 4 highest) is the same in both cities.
* **Class 4 has a construction bump.** It rises to a peak around 2000–2005, when those sites were being cleared and built, and then falls back towards the other classes. Class 3 in Amsterdam shows the same bump earlier (~1997). The *timing* of the bump tracks the construction period. That **construction event** is what the change-point features in the next section are designed to capture.
""")

md(r"""
## 5.7 Per-pixel features

Each pixel's 42-year series (11 channels) is collapsed into a fixed-length vector:

| Group | Per channel | What it captures |
|---|---|---|
| Level & spread | mean, std | Material brightness / variability |
| Trend | linear slope, late (2004+) minus early (≤2003) mean | Long-term ageing / change |
| Period means | 5 periods: 84–91, 92–99, 00–07, 08–15, 16–25 | *When* the signal changed (finer than the 2004 split) |
| Year-on-year | std and max \|change\| | Instability, construction disturbance |
| **Change-point** | best single break year, its strength, per-channel jump | **Construction event detection** for classes 3–4 |
| Largest jump | year and size of the largest year-to-year spectral change | Second construction-event estimate |
| Stability | mean std before vs after 2004 | Pre-1984 buildings are stable in both periods |

**Change-point detection.** After a 3-year running mean, we try every split year *k* and score how well a two-level step function fits:

$$\text{score}(k) = \frac{k(T-k)}{T}\,\big\lVert \bar a_{\text{after }k} - \bar a_{\text{before }k} \big\rVert^2$$

The best *k* is our estimate of **when the surface changed**, which for classes 3 and 4 is roughly the construction year.

**Spatial context.** For selected features we add the mean over all building pixels in a 5×5 window (~150 m) and an 11×11 window (~330 m). We also add the local building density.
""")

code(r"""
PERIODS = [(1984, 1991), (1992, 1999), (2000, 2007), (2008, 2015), (2016, 2025)]


def breakpoint_features(s, min_seg=3):
    # Best single change-point per pixel (two-segment mean fit scanned over all split years).
    n_pix, T, C = s.shape
    cs = np.cumsum(s, axis=1); total = cs[:, -1, :]
    best_score = np.full(n_pix, -np.inf, dtype=np.float32)
    best_k = np.zeros(n_pix, dtype=np.int32)
    best_jump = np.zeros((n_pix, C), dtype=np.float32)
    for k in range(min_seg, T - min_seg + 1):
        jump = (total - cs[:, k - 1, :]) / (T - k) - cs[:, k - 1, :] / k   # mean after - mean before
        score = (k * (T - k) / T) * (jump ** 2).sum(1)
        better = score > best_score
        best_score[better], best_k[better], best_jump[better] = score[better], k, jump[better]
    return YEARS[best_k].astype(np.float32), np.sqrt(best_score), best_jump


def pixel_features(series):
    # Collapse (pixels, years, channels) into a per-pixel feature table.
    f = {}
    t = np.arange(len(YEARS), dtype=np.float32); t -= t.mean()
    mean, std = series.mean(1), series.std(1)
    slope = (series * t[None, :, None]).sum(1) / (t ** 2).sum()
    early, late = series[:, YEARS <= 2003].mean(1), series[:, YEARS >= 2004].mean(1)
    d = np.diff(series, axis=1)
    for j, ch in enumerate(CHANNELS):
        f[f'{ch}_mean'], f[f'{ch}_std'] = mean[:, j], std[:, j]
        f[f'{ch}_slope'] = slope[:, j]
        f[f'{ch}_late_minus_early'] = late[:, j] - early[:, j]
        f[f'{ch}_dstd'] = d[:, :, j].std(1)
        f[f'{ch}_dmaxabs'] = np.abs(d[:, :, j]).max(1)
        for a, b in PERIODS:
            f[f'{ch}_p{a}'] = series[:, (YEARS >= a) & (YEARS <= b), j].mean(1)

    # 3-year running mean before change detection (suppresses single-scene noise)
    smooth = series.copy()
    smooth[:, 1:-1] = (series[:, :-2] + series[:, 1:-1] + series[:, 2:]) / 3
    f['break_year'], f['break_strength'], jump = breakpoint_features(smooth)
    for j, ch in enumerate(CHANNELS):
        f[f'{ch}_break_jump'] = jump[:, j]
    dnorm = np.sqrt((np.diff(smooth, axis=1) ** 2).sum(2))
    f['maxjump_year'] = YEARS[1:][dnorm.argmax(1)].astype(np.float32)
    f['maxjump_size'] = dnorm.max(1)
    f['std_early'] = series[:, YEARS <= 2003].std(1).mean(1)
    f['std_late'] = series[:, YEARS >= 2004].std(1).mean(1)
    return pd.DataFrame(f)


def spatial_context(feat, px, py, cols, radii=(2, 5)):
    # Mean of `cols` over building pixels within a (2r+1)x(2r+1) window, plus building density.
    gx, gy = px - px.min(), py - py.min()
    H, W = gy.max() + 1, gx.max() + 1
    mask = np.zeros((H, W), np.float32); mask[gy, gx] = 1
    out = {}
    for r in radii:
        size = 2 * r + 1
        cnt = uniform_filter(mask, size, mode='constant')
        out[f'ctx{r}_density'] = cnt[gy, gx]
        for c in cols:
            g = np.zeros((H, W), np.float32); g[gy, gx] = feat[c].values
            out[f'ctx{r}_{c}'] = (uniform_filter(g, size, mode='constant') / np.maximum(cnt, 1e-6))[gy, gx]
    return pd.DataFrame(out)


def build_features(city):
    # Final per-pixel table: city-relative time-series features + neighbourhood context.
    path = CACHE_DIR / f'features_{city}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    rel = pixel_features(ANOM[city])
    ctx_cols = [c for c in rel.columns if c.endswith('_mean') or c in ('break_year', 'break_strength')]
    ctx = spatial_context(rel, DATA[city]['px'], DATA[city]['py'], ctx_cols)
    feat = pd.concat([rel, ctx], axis=1)
    feat.to_parquet(path)
    return feat


t0 = time.time()
FEAT = {city: build_features(city) for city in DATA}
print(f'Features built in {time.time() - t0:.0f}s')
for city, f in FEAT.items():
    print(f'{city:10s} {f.shape[0]:,} pixels x {f.shape[1]} features   NaN: {int(f.isna().sum().sum())}')
""")

md(r"""
### Does the change-point year track the construction year?

If the break detector works, the detected `break_year` should follow the true construction year for buildings built during the satellite record (1984 onwards).
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, city in zip(axes, DATA):
    w, by, bs = DATA[city]['wmy'], FEAT[city]['break_year'], FEAT[city]['break_strength']
    strong = (w >= 1984) & (bs > np.percentile(bs, 50))
    ax.hexbin(w[strong], by[strong], gridsize=35, cmap='viridis', mincnt=1)
    ax.plot([1984, 2021], [1984, 2021], 'r--', lw=1, label='perfect agreement')
    r = np.corrcoef(w[strong], by[strong])[0, 1]
    ax.set_title(f'{city}: detected break vs construction year\n(post-1984 buildings, strongest 50% of breaks; r = {r:.2f})',
                 fontweight='bold')
    ax.set_xlabel('weighted_mean_year'); ax.set_ylabel('detected break_year'); ax.legend()
plt.tight_layout(); plt.show()
""")

md(r"""
**Observation:** for buildings constructed after 1984, the detected break year follows the true construction year along the diagonal in **both** cities (r ≈ 0.5). The detector is unsupervised: it uses no labels, so it works the same way in any city. It is noisy, but it gives the model a direct, city-independent estimate of *when* something was built, which is exactly what separates classes 3 and 4.
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part C — Two-stage transfer model

| Stage | Data | Model | Output |
|---|---|---|---|
| **1a** | Madrid (all labels) | LightGBM age-class **classifier** | Class scores for any city |
| **1b** | Madrid (all labels) | LightGBM **construction-year regressor** (target = `weighted_mean_year`) | A continuous "timeline" score |
| **2** | *n* labelled Amsterdam pixels per class | Methods compared below; the winner **continues boosting the Madrid classifier** on the Amsterdam samples | Amsterdam model |

### Final input: per-city robust scaling

Each feature is scaled by **its own city's** median and IQR. This uses only *unlabelled* pixels, so it is legitimate for a new city: you always have its imagery, just not its labels. It lines up the two cities' feature distributions before any label is used.
""")

code(r"""
def city_scale(df):
    # Robust z-score of every feature against its own city (unsupervised, no labels used).
    return (df - df.median()) / (df.quantile(.75) - df.quantile(.25) + 1e-9)

X  = {city: city_scale(FEAT[city]) for city in DATA}
FEATURE_NAMES = list(X['Madrid'].columns)
Xm, ym = X['Madrid'].values.astype(np.float32), Y['Madrid']
Xa, ya = X['Amsterdam'].values.astype(np.float32), Y['Amsterdam']

# LightGBM settings for Stage 1 (moderate size; balanced classes to counter class-2 dominance)
STAGE1_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.5, class_weight='balanced',
                     n_jobs=N_JOBS, verbose=-1, random_state=SEED)
print('Madrid', Xm.shape, '| Amsterdam', Xa.shape)
""")

md(r"""
## 5.8 Stage 1 — Madrid cross-validation

We report two cross-validation schemes:

* **Random stratified K-fold**: same protocol as Notebook 4, so the numbers can be compared directly.
* **Spatial block K-fold**: pixels are grouped into 1 km × 1 km blocks and whole blocks are held out. Section 5.5 showed neighbours are strongly correlated, so random folds put near-copies of test pixels into training. Spatial folds give a more honest estimate of performance on unseen areas.
""")

code(r"""
def spatial_blocks(px, py, block_m=1000):
    # Id of the square block (default 1 km) that each pixel falls in.
    cells = int(round(block_m / 30))
    return ((px - px.min()) // cells).astype(np.int64) * 100_000 + (py - py.min()) // cells


def run_cv(splitter, groups=None, label=''):
    f1s, oof = [], np.zeros(len(ym), dtype=int)
    for k, (tr, va) in enumerate(splitter.split(Xm, ym, groups), 1):
        m = lgb.LGBMClassifier(**STAGE1_PARAMS).fit(Xm[tr], ym[tr])
        oof[va] = m.predict(Xm[va])
        f1s.append(f1_score(ym[va], oof[va], average='macro'))
        print(f'  {label} fold {k}: macro-F1 = {f1s[-1]:.4f}')
    print(f'  {label}: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n')
    return np.array(f1s), oof


t0 = time.time()
cv_results = {}
for r in range(N_REPEATS):
    f, oof_random = run_cv(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r), label=f'random r{r}')
    cv_results.setdefault('random', []).extend(f)
f, oof_spatial = run_cv(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED),
                        groups=spatial_blocks(DATA['Madrid']['px'], DATA['Madrid']['py']), label='spatial')
cv_results['spatial'] = list(f)
print(f'CV done in {time.time() - t0:.0f}s')
""")

code(r"""
def plot_cm(ax, y_true, y_pred, title):
    # Row-normalised confusion matrix (each row = recall of that true class).
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4]).astype(float)
    cm /= cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm[i, j] > .6 else 'black')
    ax.set_xticks(range(4)); ax.set_xticklabels([f'C{k}' for k in range(1, 5)])
    ax.set_yticks(range(4)); ax.set_yticklabels([f'C{k}' for k in range(1, 5)])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title, fontweight='bold')


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_cm(axes[0], ym, oof_random,
        f"Madrid random CV  F1 = {np.mean(cv_results['random']):.3f} ± {np.std(cv_results['random']):.3f}")
plot_cm(axes[1], ym, oof_spatial,
        f"Madrid spatial CV  F1 = {np.mean(cv_results['spatial']):.3f} ± {np.std(cv_results['spatial']):.3f}")
plt.tight_layout(); plt.show()
""")

md(r"""
## 5.9 Stage 1 — final Madrid models

We train on all of Madrid:

* **Classifier (1a):** the model that gets transferred to Amsterdam.
* **Construction-year regressor (1b):** predicts `weighted_mean_year` directly. Unlike class labels, a *year* means the same thing in every city, so this is a city-agnostic "timeline". Stage 2 can then learn where Amsterdam's own class boundaries (e.g. 1945) fall on that timeline.
""")

code(r"""
t0 = time.time()
stage1_clf = lgb.LGBMClassifier(**STAGE1_PARAMS).fit(Xm, ym)
stage1_reg = lgb.LGBMRegressor(**{**STAGE1_PARAMS, 'class_weight': None}).fit(Xm, DATA['Madrid']['wmy'])
print(f'Stage 1 models trained in {time.time() - t0:.0f}s')

stage1_clf.booster_.save_model(str(MODEL_DIR / 'stage1_madrid_classifier.txt'))
stage1_reg.booster_.save_model(str(MODEL_DIR / 'stage1_madrid_year_regressor.txt'))
print('Saved to', MODEL_DIR.resolve())

# Outputs of the Madrid models on Amsterdam (no Amsterdam labels involved)
ams_raw   = stage1_clf.predict_proba(Xa, raw_score=True)    # class scores (log-odds)
ams_proba = stage1_clf.predict_proba(Xa)
ams_year  = stage1_reg.predict(Xa)                          # predicted construction year

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for k in range(1, 5):
    axes[0].hist(ams_year[ya == k], bins=60, alpha=.55, density=True, color=CLASS_COLORS[k], label=f'Class {k}')
axes[0].set_title('Madrid year-regressor applied to Amsterdam\n(distribution of predicted year per true class)',
                  fontweight='bold')
axes[0].set_xlabel('Predicted construction year'); axes[0].legend()
axes[1].scatter(DATA['Amsterdam']['wmy'], ams_year, s=1, alpha=.15)
axes[1].plot([1850, 2020], [1850, 2020], 'r--', lw=1)
axes[1].set_xlabel('True weighted_mean_year'); axes[1].set_ylabel('Predicted year')
axes[1].set_title(f"Amsterdam: predicted vs true year (MAE {mean_absolute_error(DATA['Amsterdam']['wmy'], ams_year):.1f} yr, "
                  f"r = {np.corrcoef(DATA['Amsterdam']['wmy'], ams_year)[0, 1]:.2f})", fontweight='bold')
plt.tight_layout(); plt.show()
""")

md(r"""
## 5.10 Zero-shot transfer: Madrid model on Amsterdam with no Amsterdam labels
""")

code(r"""
pred_zero = ams_proba.argmax(1) + 1
f1_zero = f1_score(ya, pred_zero, average='macro')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
plot_cm(axes[0], ya, pred_zero, f'Amsterdam zero-shot  F1 = {f1_zero:.3f}  (baseline 0.343)')
w = .38
axes[1].bar(np.arange(1, 5) - w / 2, np.bincount(ya)[1:], w, label='True', color='grey')
axes[1].bar(np.arange(1, 5) + w / 2, np.bincount(pred_zero, minlength=5)[1:], w, label='Predicted', color='#fd8d3c')
axes[1].set_xticks(range(1, 5)); axes[1].set_xlabel('Class'); axes[1].set_ylabel('Pixels'); axes[1].legend()
axes[1].set_title('Predicted vs true class counts', fontweight='bold')
plt.tight_layout(); plt.show()
""")

md(r"""
**Observation:** the class-2 collapse is gone. Zero-shot F1 rises from 0.343 to about 0.53 **without a single Amsterdam label**, purely from the city-relative features. A smaller bias remains: class 4 is over-predicted and class 1 under-predicted. Madrid's class 1 boundary (1960) differs from Amsterdam's (1945), and Amsterdam has relatively few class-4 buildings. These are exactly the city-specific corrections Stage 2 has to learn from a handful of labels.
""")

md(r"""
## 5.11 Stage 2 — few-shot adaptation to Amsterdam

**Protocol** (the same as Notebook 4): for each `n` in 5 / 25 / 50 / 100 / 200 we draw `n` labelled pixels per class at random (`N_TRIALS` draws, fixed seed). We adapt, then score macro-F1 on **all remaining** Amsterdam pixels. Every method sees **exactly the same draws**.

| Method | Uses Madrid model? | Idea |
|---|---|---|
| `prototype` | ✗ | Notebook 4's method (nearest class mean of the support pixels), re-run on our features |
| `timeline` | ✓ (1b) | **Two-stage timeline idea:** Madrid predicts a construction *year*; a logistic regression on that one number learns Amsterdam's class cut-points |
| `calibrate` | ✓ (1a + 1b) | Logistic regression on the Madrid class scores + predicted year (re-maps Madrid's outputs onto Amsterdam classes) |
| **`finetune`** | ✓ (1a) | **Continue boosting the Madrid classifier**: start from Madrid's scores and add a few small trees fitted to the Amsterdam samples |

`finetune` keeps everything Madrid learned (the starting scores) and only learns **corrections** for Amsterdam. The number of new trees grows with `n`, and the trees are kept small (7 leaves, L2 penalty), so 5 samples per class cannot overwrite the Madrid knowledge.
""")

code(r"""
def adapt_prototype(sup):
    cls = np.unique(ya[sup])
    protos = np.stack([Xa[sup][ya[sup] == c].mean(0) for c in cls])
    return cls[((Xa[:, None] - protos[None]) ** 2).sum(2).argmin(1)]


TIMELINE_Z = StandardScaler().fit_transform(ams_year[:, None])
CALIB_Z    = StandardScaler().fit_transform(np.column_stack([ams_raw, ams_year]))

def adapt_timeline(sup):
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    return lr.fit(TIMELINE_Z[sup], ya[sup]).predict(TIMELINE_Z)


def adapt_calibrate(sup, n):
    C = 0.05 if n <= 5 else 0.3 if n <= 50 else 1.0          # stronger regularisation for fewer labels
    lr = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
    return lr.fit(CALIB_Z[sup], ya[sup]).predict(CALIB_Z)


def finetune_booster(X_sup, y_sup, init_scores, n, seed=SEED):
    # Continue boosting from the Madrid classifier's scores on the target-city support set.
    params = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=7,
                  min_data_in_leaf=max(3, n // 5), feature_fraction=0.5, lambda_l2=5.0,
                  num_threads=N_JOBS, seed=seed, verbose=-1)
    ds = lgb.Dataset(X_sup, y_sup - 1, init_score=init_scores)
    return lgb.train(params, ds, num_boost_round=int(np.clip(n, 10, 150)))


def adapt_finetune(sup, n):
    booster = finetune_booster(Xa[sup], ya[sup], ams_raw[sup], n)
    return (ams_raw + booster.predict(Xa, raw_score=True)).argmax(1) + 1


METHODS = {'prototype': lambda sup, n: adapt_prototype(sup),
           'timeline':  lambda sup, n: adapt_timeline(sup),
           'calibrate': adapt_calibrate,
           'finetune':  adapt_finetune}

rng = np.random.default_rng(SEED)
results = {m: {n: [] for n in SHOTS} for m in METHODS}
t0 = time.time()
for n in SHOTS:
    for trial in range(N_TRIALS):
        sup = np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in range(1, 5)])
        query = np.ones(len(ya), bool); query[sup] = False
        for name, fn in METHODS.items():
            pred = fn(sup, n)
            results[name][n].append(f1_score(ya[query], pred[query], average='macro'))
    print(f'n = {n:3d}  ' + '  '.join(f'{m}: {np.mean(results[m][n]):.3f}±{np.std(results[m][n]):.3f}'
                                     for m in METHODS))
print(f'\nStage 2 evaluation done in {time.time() - t0:.0f}s')
""")

md(r"""
## 5.12 Results vs the baseline
""")

code(r"""
BASELINE = {'Madrid CV': 0.618, 'zero-shot': 0.343, 5: 0.544, 25: 0.599, 50: 0.608, 100: 0.613, 200: 0.615}

table = pd.DataFrame({m: [f'{np.mean(results[m][n]):.3f} ± {np.std(results[m][n]):.3f}' for n in SHOTS]
                      for m in METHODS}, index=[f'n = {n}' for n in SHOTS]).T
table['Average F1'] = [f'{np.mean([np.mean(results[m][n]) for n in SHOTS]):.3f}' for m in METHODS]
table.loc['Notebook 4 baseline'] = [f'{BASELINE[n]:.3f}' for n in SHOTS] + [f'{np.mean([BASELINE[n] for n in SHOTS]):.3f}']
display(table)

print(f"Madrid CV (random):  {np.mean(cv_results['random']):.3f} ± {np.std(cv_results['random']):.3f}   (baseline 0.618)")
print(f"Madrid CV (spatial): {np.mean(cv_results['spatial']):.3f} ± {np.std(cv_results['spatial']):.3f}")
print(f'Amsterdam zero-shot: {f1_zero:.3f}   (baseline 0.343)')
""")

code(r"""
# Required submission plot: F1 (± std) against log2(sample size)
fig, ax = plt.subplots(figsize=(10, 5.5))
styles = {'prototype': ('#999999', 's'), 'timeline': ('#7570b3', '^'),
          'calibrate': ('#1b9e77', 'D'), 'finetune': ('#d95f02', 'o')}
x = np.log2(SHOTS)
for m, (col, mk) in styles.items():
    ax.errorbar(x, [np.mean(results[m][n]) for n in SHOTS], yerr=[np.std(results[m][n]) for n in SHOTS],
                fmt=mk + '-', color=col, capsize=4, lw=2 if m == 'finetune' else 1.2, label=m)
ax.plot(x, [BASELINE[n] for n in SHOTS], 'k--', lw=1.2, label='Notebook 4 baseline')
ax.axhline(f1_zero, color='#d95f02', ls=':', lw=1, label=f'our zero-shot ({f1_zero:.3f})')
ax.set_xticks(x); ax.set_xticklabels([f'{n}\n(log2={np.log2(n):.1f})' for n in SHOTS])
ax.set_xlabel('Labelled Amsterdam pixels per class'); ax.set_ylabel('Macro F1 on remaining Amsterdam pixels')
ax.set_title('Few-shot transfer to Amsterdam', fontweight='bold'); ax.grid(alpha=.3); ax.legend(loc='lower right')
plt.tight_layout(); plt.show()
""")

code(r"""
# Where do the remaining errors sit? Confusion matrices of the final method at 25 and 200 shots
rng_cm = np.random.default_rng(SEED)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, n in zip(axes, [25, 200]):
    sup = np.concatenate([rng_cm.choice(np.where(ya == c)[0], n, replace=False) for c in range(1, 5)])
    query = np.ones(len(ya), bool); query[sup] = False
    pred = adapt_finetune(sup, n)
    plot_cm(ax, ya[query], pred[query], f'finetune, n = {n}/class  F1 = {f1_score(ya[query], pred[query], average="macro"):.3f}')
plt.tight_layout(); plt.show()
""")

md(r"""
## 5.13 What does the Madrid model rely on?
""")

code(r"""
imp = pd.Series(stage1_clf.booster_.feature_importance('gain'), index=FEATURE_NAMES)
imp = (imp / imp.sum()).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 7))
top = imp.head(25)[::-1]
ax.barh(top.index, top.values, color=['#d95f02' if c.startswith('ctx') else '#1b9e77' for c in top.index])
ax.set_xlabel('Share of total gain'); ax.set_title('Top 25 features (orange = neighbourhood context)', fontweight='bold')
plt.tight_layout(); plt.show()
print(f"Share of gain from context features: {imp[[c for c in imp.index if c.startswith('ctx')]].sum():.1%}")
print(f"Share of gain from change-point features: {imp[[c for c in imp.index if 'break' in c or 'maxjump' in c]].sum():.1%}")
""")

md(r"""
## 5.14 Reusable transfer procedure (submission deliverable)

`adapt_to_city(n)` is the parameterised Stage 2 required by the challenge. It:

1. loads the saved Madrid classifier,
2. draws `n` labelled pixels per class with a fixed seed,
3. continues boosting on them, and
4. returns the adapted model plus its F1 on the held-out pixels.

To run it on another city, build its features with `build_features` and pass its own `X`/`y`.
""")

code(r"""
def adapt_to_city(n, X_city=Xa, y_city=ya, seed=SEED,
                  stage1_path=MODEL_DIR / 'stage1_madrid_classifier.txt'):
    # Few-shot adaptation of the saved Madrid model to a new city with n labelled pixels per class.
    madrid = lgb.Booster(model_file=str(stage1_path))
    rng = np.random.default_rng(seed)
    sup = np.concatenate([rng.choice(np.where(y_city == c)[0], n, replace=False) for c in range(1, 5)])
    query = np.ones(len(y_city), bool); query[sup] = False
    base = madrid.predict(X_city, raw_score=True)
    booster = finetune_booster(X_city[sup], y_city[sup], base[sup], n, seed=seed)
    pred = (base + booster.predict(X_city, raw_score=True)).argmax(1) + 1
    f1 = f1_score(y_city[query], pred[query], average='macro')
    return booster, f1


for n in SHOTS:
    booster, f1 = adapt_to_city(n)
    booster.save_model(str(MODEL_DIR / f'stage2_amsterdam_n{n}.txt'))
    print(f'n = {n:3d}: held-out macro-F1 = {f1:.3f}  -> saved models/stage2_amsterdam_n{n}.txt')
""")

md(r"""
## 5.15 Summary & next steps

**What we found (Part A)**
* The zero-shot collapse is a **brightness offset**: Amsterdam is darker in every band, but the within-city age ordering is identical.
* City-wide year-to-year shifts (atmosphere, sensor) are noise shared by every pixel.
* Building age is **strongly spatially clustered** in both cities.
* Construction leaves a **detectable event** in the time series (break year tracks construction year, r ≈ 0.5, with no labels).

**What we built (Parts B & C)**
* City-year anomaly series → change-point, trend and period features → neighbourhood context → per-city robust scaling.
* Stage 1: LightGBM classifier + construction-year regressor on Madrid.
* Stage 2: **continued boosting** of the Madrid model on the Amsterdam samples, which actually reuses Madrid's knowledge (the baseline does not).

**Notes and caveats**
* The "timeline only" variant (predict a year, then learn Amsterdam cut-points) underperforms on its own. A single number throws away too much information to separate the four classes. The predicted year is more useful as one extra input (`calibrate`).
* The random-fold Madrid CV is optimistic because of spatial autocorrelation. The spatial-block CV is the more honest estimate.

**Next steps to try**
* Self-training: pseudo-label confident unlabelled Amsterdam pixels and add them to Stage 2.
* Ensembling `finetune` with `calibrate`/`prototype` for the 5-shot case.
* Spatial smoothing of predicted probabilities (age is clustered).
""")

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
OUT = '5-Our_Model (DSRH2026).ipynb'
nbf.write(nb, OUT)
print('Wrote', OUT, 'with', len(cells), 'cells')
