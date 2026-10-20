"""Shared helpers for Notebook 5: memory-light loading of the Landsat parquet files.

Each city is loaded into a dense "cube" of shape (n_pixels, n_years, n_bands) instead of
a 3M-row DataFrame. String columns (pixel_id, scene_id_N, dataset_N) are never read, which
keeps peak memory well under 2 GB on an 8 GB machine.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
YEARS = np.arange(1984, 2026)
BLUE_MAX = 15_000          # haze threshold from Notebook 2
CLASS_EVENTS = {'Amsterdam': 1945, 'Madrid': 1960}
FILES = {'Madrid': 'madrid_train.parquet', 'Amsterdam': 'amsterdam_data.parquet'}


def find_data_dir():
    return next(p for p in [Path('data'), Path('../data')]
                if (p / 'madrid_train.parquet').exists())


def age_class(year, city):
    """Bin weighted_mean_year into classes 1-4 with the city-specific 1/2 boundary."""
    bins = [-np.inf, CLASS_EVENTS[city], 1984, 2004, np.inf]
    return pd.cut(year, bins=bins, labels=[1, 2, 3, 4]).astype(int)


def load_cube(city, data_dir=None, cache=True):
    """Return a dict with the per-pixel annual reflectance cube and pixel metadata.

    cube[i, t, b]  median of the QA-valid, haze-free observations of pixel i in year YEARS[t]
                   for band b (NaN if no valid observation that year).
    n_obs[i, t]    number of valid observations that went into the median (0-3).
    doy[i, t]      mean day-of-year of those observations.
    """
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    cache_path = data_dir / 'cache' / f'cube_{city}.npz'
    if cache and cache_path.exists():
        z = np.load(cache_path)
        out = {k: z[k] for k in z.files}
        out['city'] = city
        return out

    cols = ['year', 'px_key', 'py_key', 'pixel_x', 'pixel_y', 'coverage', 'weighted_mean_year']
    for s in (1, 2, 3):
        cols += [f'{b}_{s}' for b in BANDS] + [f'qa_valid_{s}', f'doy_{s}']
    t = pq.read_table(data_dir / FILES[city], columns=cols)

    px = t['px_key'].to_numpy(); py = t['py_key'].to_numpy()
    keys = px.astype(np.int64) * 1_000_000 + py
    ukeys, pix_idx = np.unique(keys, return_inverse=True)
    yr_idx = t['year'].to_numpy() - YEARS[0]
    n_pix, n_yr = len(ukeys), len(YEARS)

    # Stack the 3 observation slots: (rows, 3, bands)
    obs = np.stack([np.stack([t[f'{b}_{s}'].to_numpy(zero_copy_only=False).astype(np.float32)
                              for b in BANDS], axis=1) for s in (1, 2, 3)], axis=1)
    valid = np.stack([t[f'qa_valid_{s}'].to_numpy(zero_copy_only=False) for s in (1, 2, 3)], axis=1)
    valid = np.nan_to_num(valid.astype(float), nan=0).astype(bool)
    valid &= np.isfinite(obs).all(axis=2)
    valid &= (obs > 0).all(axis=2) & (obs < 65535).all(axis=2)   # corrupt zero / saturated
    valid &= obs[:, :, 0] <= BLUE_MAX                             # haze
    obs[~valid] = np.nan
    doy = np.stack([t[f'doy_{s}'].to_numpy(zero_copy_only=False).astype(np.float32)
                    for s in (1, 2, 3)], axis=1)
    doy[~valid] = np.nan

    with np.errstate(all='ignore'):
        med = np.nanmedian(obs, axis=1)            # (rows, bands)
        mdoy = np.nanmean(doy, axis=1)
    cube = np.full((n_pix, n_yr, len(BANDS)), np.nan, dtype=np.float32)
    cube[pix_idx, yr_idx] = med
    n_obs = np.zeros((n_pix, n_yr), dtype=np.int8)
    n_obs[pix_idx, yr_idx] = valid.sum(axis=1)
    doy_c = np.full((n_pix, n_yr), np.nan, dtype=np.float32)
    doy_c[pix_idx, yr_idx] = mdoy

    # Per-pixel constants (first occurrence)
    first = np.unique(pix_idx, return_index=True)[1]
    out = dict(
        cube=cube, n_obs=n_obs, doy=doy_c,
        px=px[first].astype(np.int32), py=py[first].astype(np.int32),
        x=t['pixel_x'].to_numpy()[first].astype(np.float64),
        y=t['pixel_y'].to_numpy()[first].astype(np.float64),
        coverage=t['coverage'].to_numpy()[first].astype(np.float32),
        wmy=t['weighted_mean_year'].to_numpy()[first].astype(np.float32),
    )
    if cache:
        cache_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache_path, **out)
    out['city'] = city
    return out


# ════════════════════════════════════════════════════════════════════════════
# Time-series preparation
# ════════════════════════════════════════════════════════════════════════════
INDICES = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']


def fill_gaps(cube):
    """Linear interpolation along the year axis; edges take the nearest observed value."""
    c = cube.copy()
    n_pix, n_yr, n_b = c.shape
    t = np.arange(n_yr)
    valid = np.isfinite(c[:, :, 0])
    prev = np.where(valid, t, -1); prev = np.maximum.accumulate(prev, axis=1)
    nxt = np.where(valid, t, n_yr); nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    prev_c = np.where(prev < 0, nxt, prev); nxt_c = np.where(nxt >= n_yr, prev_c, nxt)
    rows = np.arange(n_pix)[:, None]
    lo, hi = c[rows, prev_c], c[rows, nxt_c]
    w = np.where(nxt_c > prev_c, (t - prev_c) / np.maximum(nxt_c - prev_c, 1), 0.0)[:, :, None]
    filled = lo + w * (hi - lo)
    return np.where(valid[:, :, None], c, filled).astype(np.float32)


def add_indices(cube):
    """Append the 5 normalised-difference indices as extra channels."""
    B, G, R, N, S1, S2 = [cube[:, :, i] for i in range(6)]
    nd = lambda a, b: (a - b) / (a + b + 1e-6)
    idx = np.stack([nd(N, R), nd(S1, N), nd(S2, N), nd(G, S1),
                    nd(S1 + R, N + B)], axis=2)
    return np.concatenate([cube, idx.astype(np.float32)], axis=2)


def city_year_anomaly(series):
    """Express every value relative to the city in the same year:
    (x - median over pixels) / IQR over pixels, per year and channel.
    Removes city-wide offsets (climate, materials, atmosphere, sensor) and keeps
    the within-city ranking, which is the part that transfers between cities."""
    med = np.nanmedian(series, axis=0, keepdims=True)
    q75, q25 = np.nanpercentile(series, [75, 25], axis=0, keepdims=True)
    return ((series - med) / (q75 - q25 + 1e-6)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Per-pixel feature engineering
# ════════════════════════════════════════════════════════════════════════════
CHANNELS = BANDS + INDICES
PERIODS = [(1984, 1991), (1992, 1999), (2000, 2007), (2008, 2015), (2016, 2025)]


def _slope(s):
    t = np.arange(s.shape[1], dtype=np.float32); t -= t.mean()
    return (s * t[None, :, None]).sum(1) / (t ** 2).sum()


def breakpoint_features(s, min_seg=3):
    """Best single change-point per pixel, scanned over every split year.

    For each split k the score is the squared difference between the mean after and
    before k, summed over channels and weighted by the segment sizes (a CUSUM / two-
    segment fit). Returns the break year, its strength, and the per-channel jump."""
    n_pix, T, C = s.shape
    cs = np.cumsum(s, axis=1)
    total = cs[:, -1:, :]
    ks = np.arange(min_seg, T - min_seg + 1)
    best_score = np.full(n_pix, -np.inf, dtype=np.float32)
    best_k = np.zeros(n_pix, dtype=np.int32)
    best_jump = np.zeros((n_pix, C), dtype=np.float32)
    for k in ks:
        before = cs[:, k - 1, :] / k
        after = (total[:, 0, :] - cs[:, k - 1, :]) / (T - k)
        jump = after - before
        score = (k * (T - k) / T) * (jump ** 2).sum(1)
        better = score > best_score
        best_score[better] = score[better]; best_k[better] = k
        best_jump[better] = jump[better]
    return YEARS[best_k].astype(np.float32), np.sqrt(best_score), best_jump


def pixel_features(series, prefix=''):
    """Collapse a (pixels, years, channels) series into a per-pixel feature table."""
    f = {}
    yrs = YEARS
    mean, std = series.mean(1), series.std(1)
    slope = _slope(series)
    early = series[:, yrs <= 2003].mean(1); late = series[:, yrs >= 2004].mean(1)
    d = np.diff(series, axis=1)
    for j, ch in enumerate(CHANNELS):
        f[f'{ch}_mean'] = mean[:, j]; f[f'{ch}_std'] = std[:, j]
        f[f'{ch}_slope'] = slope[:, j]
        f[f'{ch}_late_minus_early'] = late[:, j] - early[:, j]
        f[f'{ch}_dstd'] = d[:, :, j].std(1)
        f[f'{ch}_dmaxabs'] = np.abs(d[:, :, j]).max(1)
        for a, b in PERIODS:
            f[f'{ch}_p{a}'] = series[:, (yrs >= a) & (yrs <= b), j].mean(1)
    # Change-point on all channels jointly (smoothed with a 3-yr running mean first)
    k = np.ones(3) / 3
    smooth = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, series)
    smooth[:, 0] = series[:, 0]; smooth[:, -1] = series[:, -1]
    byear, bstrength, bjump = breakpoint_features(smooth)
    f['break_year'] = byear; f['break_strength'] = bstrength
    for j, ch in enumerate(CHANNELS):
        f[f'{ch}_break_jump'] = bjump[:, j]
    # Year of the largest year-on-year change in the overall spectral vector
    dnorm = np.sqrt((np.diff(smooth, axis=1) ** 2).sum(2))
    f['maxjump_year'] = YEARS[1:][dnorm.argmax(1)].astype(np.float32)
    f['maxjump_size'] = dnorm.max(1)
    # Stability before / after 2004 (pre-1984 buildings stay stable in both)
    f['std_early'] = series[:, yrs <= 2003].std(1).mean(1)
    f['std_late'] = series[:, yrs >= 2004].std(1).mean(1)
    return pd.DataFrame({prefix + k: v for k, v in f.items()})


def spatial_context(feat, px, py, cols, radii=(2, 5)):
    """Neighbourhood means of selected features over building pixels within a square
    window of radius r grid cells (r=2 -> 150 m, r=5 -> 330 m). Building age is spatially
    clustered (whole neighbourhoods are built together), so context carries signal."""
    from scipy.ndimage import uniform_filter
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


def build_features(d, context_cols=None):
    """Full feature table for one city cube (city-relative, transfer-oriented)."""
    series = add_indices(fill_gaps(d['cube']))
    anom = city_year_anomaly(series)
    feat = pixel_features(anom)
    if context_cols is None:
        context_cols = [c for c in feat.columns
                        if c.endswith('_mean') or c in ('break_year', 'break_strength')]
    ctx = spatial_context(feat, d['px'], d['py'], context_cols)
    return pd.concat([feat, ctx], axis=1)


def spatial_blocks(px, py, block_m=1000):
    """Group id of the square block (default 1 km) each pixel falls in. Used for spatial
    cross-validation so neighbouring pixels never sit on both sides of a split."""
    cells = int(round(block_m / 30))
    bx, by = (px - px.min()) // cells, (py - py.min()) // cells
    return (bx.astype(np.int64) * 100_000 + by).astype(np.int64)


def build_raw_features(d):
    """Absolute-reflectance features in the style of Notebook 3 (for comparison)."""
    series = add_indices(fill_gaps(d['cube']))
    return pixel_features(series)
