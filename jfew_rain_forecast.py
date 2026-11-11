#!/usr/bin/env python3
"""
J-FEW · PRAKIRAAN CURAH HUJAN TAHUN DEPAN + ANALISIS FREKUENSI — jfew_rain_forecast.py (v1 · 11 Nov 2026)
--------------------------------------------------------------------------------------------------------
Data masuk (dari web app J-FEW, action "rain_input"):
  • seri bulanan gabungan 1998–sekarang (ombrometer bila valid, selain itu IMERG terkoreksi) + ombrometer + IMERG
  • data HARIAN ombrometer (sheet "Ombrometer Harian"/"Harian": tanggal, hujan mm, durasi jam, slippery jam)
Keluaran (action "rain_fc" → sheet JFEW_RAIN_FC → peta & "Dashboard Curah Hujan"):
  A. Prakiraan bulanan TAHUN DEPAN (Jan–Des) per metode:
     Klimatologi · Persentil P50/P60/P70/P80 · ARIMA (anomali) · SARIMA · Holt-Winters ·
     Regresi ENSO+IOD · Analog ENSO+IOD · Hibrida ECMWF SEAS5 · Ensemble berbobot (bobot = 1/MAE hindcast)
     + uji hindcast (tiap tahun 2015…tahun lalu, titik awal = bulan yang sama dengan data terakhir)
  B. Analisis frekuensi (Normal, Log Normal, Gumbel, Log-Pearson III; uji Smirnov-Kolmogorov & Chi-kuadrat, α 5 %):
     • hujan BULANAN maksimum tahunan (ombrometer)
     • hujan HARIAN maksimum tahunan R24 (IMERG harian 1998–sekarang, dikoreksi ke ombrometer harian — quantile ratio ekor atas)
  C. Hujan harian → slippery (tabel ambang)
ENSO = anomali Niño 3.4 (NOAA PSL) · IOD = Dipole Mode Index (NOAA PSL, HadISST) · SEAS5 = Open-Meteo Seasonal API.
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT, JFEW_WEBAPP_URL, JFEW_BURN_TOKEN · opsional JFEW_LOCAL_INPUT (uji lokal), NO_EE=1
"""
import calendar
import datetime as dt
import json
import math
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy import stats as SS

warnings.filterwarnings("ignore")
LAT, LON = -3.903238, 115.018279
PCTS = [50, 60, 70, 80]
T_RET = [2, 5, 10, 25, 50, 100]


# ---------------------------------------------------------------- input
def get_input():
    if os.environ.get("JFEW_LOCAL_INPUT"):
        return json.load(open(os.environ["JFEW_LOCAL_INPUT"]))
    import jfew_burn_job as J
    return J.web({"action": "rain_input"})


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "J-FEW rain forecast"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def psl_series(url):
    """Format NOAA PSL: baris 'tahun v1..v12', nilai hilang -99.99 / -9999."""
    out = {}
    for ln in fetch(url).splitlines():
        p = ln.split()
        if len(p) == 13 and p[0].isdigit() and 1850 < int(p[0]) < 2200:
            for m, v in enumerate(p[1:], 1):
                try:
                    x = float(v)
                except ValueError:
                    continue
                if x > -90:
                    out[pd.Period("%s-%02d" % (p[0], m), "M")] = x
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------- metode prakiraan
def clim_of(tr):
    return tr.groupby(tr.index.month).mean()


def months_ahead(tr, target):
    """indeks bulan (1..h) dari akhir data sampai Jan–Des tahun target"""
    last = tr.index[-1]
    start = pd.Period("%d-01" % target, "M")
    h0 = (start - last).n
    return h0, h0 + 11


def m_clim(tr, target, **k):
    c = clim_of(tr)
    return np.array([c[m] for m in range(1, 13)])


def m_pct(q):
    def f(tr, target, **k):
        c = tr.groupby(tr.index.month).quantile(q / 100)
        return np.array([c[m] for m in range(1, 13)])
    return f


def _tail(fc, tr, target):
    a, b = months_ahead(tr, target)
    return np.clip(np.asarray(fc)[a - 1:b], 0, None)


def m_arima(tr, target, **k):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    c = clim_of(tr)
    an = tr - np.array([c[p.month] for p in tr.index])
    a, b = months_ahead(tr, target)
    r = SARIMAX(an.values, order=(1, 0, 1)).fit(disp=False)
    f = r.forecast(b)
    base = np.array([c[(tr.index[-1] + i).month] for i in range(1, b + 1)])
    return _tail(base + f, tr, target)


def m_sarima(tr, target, **k):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    a, b = months_ahead(tr, target)
    best = None
    for o in [(1, 0, 0), (1, 0, 1), (0, 0, 1)]:
        for so in [(1, 0, 1), (0, 1, 1), (1, 1, 1)]:
            try:
                r = SARIMAX(np.log1p(tr.values), order=o, seasonal_order=so + (12,), trend="c" if so[1] == 0 else "n").fit(disp=False)
                if best is None or r.aic < best.aic:
                    best = r
            except Exception:  # noqa: BLE001
                pass
    return _tail(np.expm1(best.forecast(b)), tr, target)


def m_hw(tr, target, **k):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    a, b = months_ahead(tr, target)
    r = ExponentialSmoothing(tr.values, trend=None, seasonal="add", seasonal_periods=12,
                             initialization_method="estimated").fit(smoothing_level=0.05, smoothing_seasonal=0.05, optimized=False)
    return _tail(r.forecast(b), tr, target)


def predictors(idx, enso, iod, at):
    """rata-rata 3 bulan terakhir yang tersedia s.d. bulan 'at' (tanpa mengintip masa depan)"""
    def v(s):
        s = s[s.index <= at].tail(3)
        return float(s.mean()) if len(s) else np.nan
    return v(enso), v(iod)


def m_reg(tr, target, enso=None, iod=None, **k):
    """rasio hujan/klimatologi per musim (basah Nov–Apr, kering Mei–Okt) = 1 + b1·Niño3.4 + b2·DMI (prediktor saat titik awal)"""
    if enso is None or iod is None or not len(enso) or not len(iod):
        return None
    c = clim_of(tr)
    om = tr.index[-1].month
    X, Y, S = [], [], []
    for y in range(tr.index[0].year, target - 1):
        at = pd.Period("%d-%02d" % (y, om), "M")
        if at > tr.index[-1]:
            break
        n, d = predictors(None, enso, iod, at)
        if np.isnan(n) or np.isnan(d):
            continue
        for m in range(1, 13):
            p = pd.Period("%d-%02d" % (y + 1, m), "M")
            if p in tr.index and p <= tr.index[-1] and c[m] > 0:
                X.append([1, n, d]); Y.append(tr[p] / c[m]); S.append(m in (11, 12, 1, 2, 3, 4))
    if len(Y) < 36:
        return None
    X, Y, S = np.array(X), np.array(Y), np.array(S)
    n0, d0 = predictors(None, enso, iod, tr.index[-1])
    if np.isnan(n0) or np.isnan(d0):
        return None
    out = []
    for m in range(1, 13):
        w = S == (m in (11, 12, 1, 2, 3, 4))
        beta = np.linalg.lstsq(X[w], Y[w], rcond=None)[0]
        r = float(np.clip(beta @ [1, n0, d0], 0.3, 2.5))
        out.append(c[m] * r)
    return np.array(out)


def m_analog(tr, target, enso=None, iod=None, k=5, **kw):
    if enso is None or iod is None or not len(enso) or not len(iod):
        return None
    c = clim_of(tr)
    om = tr.index[-1].month
    n0, d0 = predictors(None, enso, iod, tr.index[-1])
    if np.isnan(n0) or np.isnan(d0):
        return None
    sn, sd = float(enso.std()) or 1, float(iod.std()) or 1
    cand = []
    for y in range(tr.index[0].year, target - 1):
        at = pd.Period("%d-%02d" % (y, om), "M")
        n, d = predictors(None, enso, iod, at)
        ps = [pd.Period("%d-%02d" % (y + 1, m), "M") for m in range(1, 13)]
        if np.isnan(n) or any(p not in tr.index or p > tr.index[-1] for p in ps):
            continue
        cand.append((math.hypot((n - n0) / sn, (d - d0) / sd), y))
    cand.sort()
    use = [y for _, y in cand[:k]]
    if len(use) < 3:
        return None
    rat = np.array([[tr[pd.Period("%d-%02d" % (y + 1, m), "M")] / c[m] for m in range(1, 13)] for y in use])
    return np.array([c[m] for m in range(1, 13)]) * rat.mean(axis=0), use


def seas5(target, clim_sat):
    """Hibrida ECMWF SEAS5 (Open-Meteo Seasonal): bulan tahun target yang tercakup penuh; rasio terhadap klimatologi IMERG"""
    for model, days in [("ecmwf_seasonal_seamless", 215), ("ecmwf_seas5", 183)]:
        try:
            j = json.loads(fetch("https://seasonal-api.open-meteo.com/v1/seasonal?latitude=%s&longitude=%s&daily=precipitation_sum&forecast_days=%d&models=%s"
                                 % (LAT, LON, days, model), 90))
            d = j.get("daily") or {}
            T = d.get("time") or []
            mem = [v for k, v in d.items() if k.startswith("precipitation_sum")]
            if not T or not mem:
                continue
            arr = np.array([[np.nan if x is None else x for x in m] for m in mem], dtype=float)
            mean = np.nanmean(arr, axis=0)
            df = pd.Series(mean, index=pd.to_datetime(T))
            out = {}
            for m in range(1, 13):
                s = df[(df.index.year == target) & (df.index.month == m)]
                if len(s) == calendar.monthrange(target, m)[1] and not s.isna().any():
                    out[m] = float(s.sum())
            return out, model
        except Exception as ex:  # noqa: BLE001
            print("SEAS5", model, ex)
    return {}, ""


# ---------------------------------------------------------------- analisis frekuensi
def freq(x):
    x = np.array([v for v in x if v is not None and v > 0], dtype=float)
    n = len(x)
    if n < 8:
        return None
    m, s = x.mean(), x.std(ddof=1)
    lx = np.log10(x); ml, sl = lx.mean(), lx.std(ddof=1)
    cs = SS.skew(lx, bias=False)
    z = lambda T: SS.norm.ppf(1 - 1 / T)                                     # noqa: E731
    def kt(T):                                                                # faktor frekuensi Pearson III (Kite / Wilson-Hilferty)
        zz, k = z(T), cs / 6
        return zz + (zz * zz - 1) * k + (zz ** 3 - 6 * zz) * k * k / 3 - (zz * zz - 1) * k ** 3 + zz * k ** 4 + k ** 5 / 3
    yT = lambda T: -math.log(-math.log(1 - 1 / T))                            # noqa: E731
    D = {
        "Normal": (lambda T: m + z(T) * s, lambda v: SS.norm.cdf((v - m) / s)),
        "Log Normal": (lambda T: 10 ** (ml + z(T) * sl), lambda v: SS.norm.cdf((math.log10(v) - ml) / sl)),
        "Gumbel": (lambda T: m + (-(math.sqrt(6) / math.pi) * (0.5772 + math.log(math.log(T / (T - 1))))) * s,
                   lambda v: math.exp(-math.exp(-((v - (m - 0.5772 * s * math.sqrt(6) / math.pi)) / (s * math.sqrt(6) / math.pi))))),
        "Log-Pearson III": (lambda T: 10 ** (ml + kt(T) * sl), None),
    }
    def lp3_cdf(v):
        lo, hi = 1.0001, 1e5
        if v <= D["Log-Pearson III"][0](lo):
            return 1 - 1 / lo
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            if D["Log-Pearson III"][0](mid) < v: lo = mid
            else: hi = mid
        return 1 - 1 / lo
    D["Log-Pearson III"] = (D["Log-Pearson III"][0], lp3_cdf)
    xs = np.sort(x)
    k = int(round(1 + 3.322 * math.log10(n)))
    dfree = max(1, k - 1 - 2)
    chi_c, ks_c = float(SS.chi2.ppf(0.95, dfree)), float(SS.ksone.ppf(1 - 0.05 / 2, n))
    res = []
    for name, (inv, cdf) in D.items():
        dmax = max(abs((i + 1) / (n + 1) - cdf(v)) for i, v in enumerate(xs))
        edges = [inv(1 / (1 - p)) for p in [j / k for j in range(1, k)]]
        cnt = np.zeros(k)
        for v in x:
            cnt[sum(1 for e in edges if v > e)] += 1
        chi = float(((cnt - n / k) ** 2 / (n / k)).sum())
        res.append({"name": name, "R": [round(float(inv(T)), 1) for T in T_RET], "ks": round(dmax, 3), "chi": round(chi, 2),
                    "ok": bool(dmax < ks_c and chi < chi_c)})
    for key in ["ks", "chi"]:
        for i, r in enumerate(sorted(res, key=lambda r: r[key])):
            r["rk_" + key] = i + 1
    for r in res:
        r["rank"] = (r["rk_ks"] + r["rk_chi"]) / 2 + (0 if r["ok"] else 10)
    res.sort(key=lambda r: r["rank"])
    return {"n": n, "mean": round(float(m), 1), "sd": round(float(s), 1), "cs_log": round(float(cs), 3), "ks_crit": round(ks_c, 3),
            "chi_crit": round(chi_c, 2), "df": dfree, "T": T_RET, "dist": res, "best": res[0]["name"], "obs": [round(float(v), 1) for v in sorted(x, reverse=True)]}


def imerg_daily(daily_dates, years):
    """IMERG harian (hari WITA) di titik ombrometer: (a) tanggal yang ada data ombrometer, (b) maksimum harian per tahun"""
    import ee
    import jfew_burn_job as J
    J.init_ee()
    col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
    pt = ee.Geometry.Point([LON, LAT])
    def day_img(d):                                                            # hari WITA = 16:00 UTC kemarin s.d. 16:00 UTC
        s = ee.Date(d).advance(-8, "hour")
        return col.filterDate(s, s.advance(1, "day")).sum().multiply(0.5).set("d", ee.Date(d).format("YYYY-MM-dd"))
    pairs = {}
    ds = sorted(daily_dates)
    for i in range(0, len(ds), 120):
        chunk = ds[i:i + 120]
        fc = ee.FeatureCollection([ee.Feature(None, {"d": d, "v": day_img(d).reduceRegion(ee.Reducer.first(), pt, 11132).values().get(0)}) for d in chunk]).getInfo()
        for f in fc["features"]:
            v = f["properties"].get("v")
            if v is not None:
                pairs[f["properties"]["d"]] = float(v)
        print("pasangan harian", len(pairs))
    amax = {}
    for y in years:
        n = (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days
        days = ee.List.sequence(0, n - 1).map(lambda k: day_img(ee.Date.fromYMD(y, 1, 1).advance(k, "day")))
        v = ee.ImageCollection.fromImages(days).max().reduceRegion(ee.Reducer.first(), pt, 11132).values().get(0).getInfo()
        if v is not None:
            amax[y] = float(v)
        print("R24 IMERG", y, amax.get(y))
    return pairs, amax


# ---------------------------------------------------------------- utama
def main():
    I = get_input()
    rows = I["months"]                                                         # [ym, gauge, imerg, comb, flag]
    S = pd.Series({pd.Period(r[0], "M"): r[3] for r in rows if r[3] is not None}).sort_index()
    G = pd.Series({pd.Period(r[0], "M"): r[1] for r in rows if r[1] is not None}).sort_index()
    SAT = pd.Series({pd.Period(r[0], "M"): r[2] for r in rows if r[2] is not None}).sort_index()
    S = S[S.index <= S.index.max()]
    last = S.index[-1]
    target = last.year + 1
    print("seri gabungan", S.index[0], "→", last, "· target", target)

    enso = iod = None
    try:
        enso = psl_series("https://psl.noaa.gov/data/correlation/nina34.anom.data")
        print("Niño3.4 s.d.", enso.index[-1])
    except Exception as ex:  # noqa: BLE001
        print("ENSO gagal:", ex)
    try:
        iod = psl_series("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data")
        print("DMI s.d.", iod.index[-1])
    except Exception as ex:  # noqa: BLE001
        print("IOD gagal:", ex)

    M = {"Klimatologi": m_clim, "ARIMA (anomali)": m_arima, "SARIMA": m_sarima, "Holt-Winters": m_hw,
         "Regresi ENSO + IOD": m_reg, "Analog ENSO + IOD": lambda tr, t, **k: (lambda r: None if r is None else r[0])(m_analog(tr, t, **k))}
    for q in PCTS:
        M["Persentil P%d%s" % (q, " (median)" if q == 50 else "")] = m_pct(q)
    kw = {"enso": enso, "iod": iod}

    # --- hindcast: tiap tahun Y (2015…target-1), titik awal = bulan 'last.month' tahun Y-1
    err = {k: [] for k in M}; ann = {k: [] for k in M}
    for Y in range(2015, target):
        origin = pd.Period("%d-%02d" % (Y - 1, last.month), "M")
        tr = S[S.index <= origin]
        obs = [S.get(pd.Period("%d-%02d" % (Y, m), "M")) for m in range(1, 13)]
        if any(o is None or (isinstance(o, float) and np.isnan(o)) for o in obs) or len(tr) < 60:
            continue
        obs = np.array(obs, dtype=float)
        for k, f in M.items():
            try:
                p = f(tr, Y, **kw)
                if p is not None:
                    err[k].append(p - obs); ann[k].append(abs(p.sum() - obs.sum()))
            except Exception as ex:  # noqa: BLE001
                print("hindcast", k, Y, ex)
    skill = {}
    for k in M:
        if err[k]:
            e = np.concatenate(err[k])
            skill[k] = {"mae": round(float(np.abs(e).mean()), 1), "rmse": round(float(np.sqrt((e ** 2).mean())), 1),
                        "ann": round(float(np.mean(ann[k])), 0), "n_years": len(err[k])}
    cmae = skill.get("Klimatologi", {}).get("mae")
    for k, v in skill.items():
        v["vs_clim_pct"] = round((cmae - v["mae"]) / cmae * 100, 1) if cmae else None

    # --- prakiraan tahun target
    F, notes = {}, {}
    for k, f in M.items():
        try:
            p = f(S, target, **kw)
            if p is not None:
                F[k] = [round(float(v), 1) for v in p]
        except Exception as ex:  # noqa: BLE001
            notes[k] = str(ex)[:120]
    an = m_analog(S, target, **kw) if enso is not None and iod is not None else None
    if an:
        notes["Analog ENSO + IOD"] = "tahun analog (tahun berikutnya dipakai): " + ", ".join(str(y) for y in an[1])
    # Hibrida SEAS5
    try:
        clim_sat = SAT.groupby(SAT.index.month).mean() if len(SAT) else None
        s5, model = seas5(target, clim_sat)
        if s5 and clim_sat is not None:
            base = F.get("Ensemble berbobot") or None
            c = clim_of(S)
            hyb = []
            for m in range(1, 13):
                if m in s5 and clim_sat.get(m, 0) > 0:
                    hyb.append(c[m] * float(np.clip(s5[m] / clim_sat[m], 0.3, 2.5)))
                else:
                    hyb.append(None)
            F["Hibrida ECMWF SEAS5"] = hyb
            notes["Hibrida ECMWF SEAS5"] = "%s · bulan %s dari SEAS5 (rasio thd klimatologi IMERG); bulan lain diisi Ensemble" % (
                model, ", ".join(calendar.month_abbr[m] for m in sorted(s5)))
    except Exception as ex:  # noqa: BLE001
        notes["Hibrida ECMWF SEAS5"] = str(ex)[:120]
    # Ensemble berbobot (metode berbasis model, bukan persentil), bobot 1/MAE hindcast
    pool = [k for k in ["Klimatologi", "ARIMA (anomali)", "SARIMA", "Holt-Winters", "Regresi ENSO + IOD", "Analog ENSO + IOD"] if k in F and k in skill]
    w = {k: 1 / skill[k]["mae"] for k in pool}
    tw = sum(w.values())
    w = {k: v / tw for k, v in w.items()}
    F["Ensemble berbobot"] = [round(sum(F[k][i] * w[k] for k in pool), 1) for i in range(12)]
    if "Hibrida ECMWF SEAS5" in F:
        F["Hibrida ECMWF SEAS5"] = [round(v, 1) if v is not None else F["Ensemble berbobot"][i] for i, v in enumerate(F["Hibrida ECMWF SEAS5"])]
    order = ["Ensemble berbobot", "Klimatologi"] + ["Persentil P%d%s" % (q, " (median)" if q == 50 else "") for q in PCTS] + \
            ["ARIMA (anomali)", "SARIMA", "Holt-Winters", "Regresi ENSO + IOD", "Analog ENSO + IOD", "Hibrida ECMWF SEAS5"]
    order = [k for k in order if k in F]

    # --- analisis frekuensi
    fr = {}
    gy = {}
    for p, v in G.items():
        gy.setdefault(p.year, []).append(v)
    fr["monthly"] = freq([max(v) for y, v in gy.items() if len(v) == 12])
    if fr["monthly"]:
        fr["monthly"]["years"] = "%d–%d" % (min(y for y, v in gy.items() if len(v) == 12), max(y for y, v in gy.items() if len(v) == 12))
    daily = I.get("daily") or []
    D = pd.DataFrame(daily, columns=["d", "r", "du", "sl"]) if daily else pd.DataFrame(columns=["d", "r", "du", "sl"])
    if os.environ.get("NO_EE") != "1" and len(D):
        try:
            years = list(range(1998, dt.date.today().year))
            pairs, amax = imerg_daily(list(D["d"]), years)
            P = D[D["d"].isin(pairs)].copy(); P["s"] = P["d"].map(pairs)
            wet = P[(P["r"] > 1) | (P["s"] > 1)]
            qs = np.arange(0.95, 0.996, 0.005)
            ratio = float(np.median([np.quantile(wet["r"], q) / max(np.quantile(wet["s"], q), 0.1) for q in qs])) if len(wet) > 100 else 1.0
            ratio = float(np.clip(ratio, 0.7, 3.0))
            r24 = freq([v * ratio for v in amax.values()])
            if r24:
                r24.update({"years": "%d–%d" % (min(amax), max(amax)), "factor": round(ratio, 3), "pairs": int(len(P)),
                            "r_daily": round(float(np.corrcoef(P["r"], P["s"])[0, 1]), 2) if len(P) > 30 else None})
                # intensitas 1 jam (Mononobe) dari R24 terpilih
                b = [d for d in r24["dist"] if d["name"] == r24["best"]][0]
                r24["mononobe_1h"] = [round(R / 24 * (24 / 1) ** (2 / 3), 1) for R in b["R"]]
            fr["r24"] = r24
        except Exception as ex:  # noqa: BLE001
            print("R24 gagal:", ex); fr["r24_error"] = str(ex)[:200]
    # peluang hujan ekstrem tahun target (dari distribusi terpilih)
    # --- hujan harian → slippery
    slip = []
    if len(D):
        D = D.astype({"r": float, "du": float, "sl": float})
        for a, b, lab in [(0, 0, "0 mm"), (0.1, 5, "0,1–5 mm"), (5, 10, "5–10 mm"), (10, 20, "10–20 mm"), (20, 40, "20–40 mm"), (40, 80, "40–80 mm"), (80, 1e9, "> 80 mm")]:
            sel = D[D["r"] == 0] if b == 0 else D[(D["r"] >= a) & (D["r"] < b)]
            if len(sel):
                slip.append({"label": lab, "n": int(len(sel)), "slip_mean": round(float(sel["sl"].mean()), 2),
                             "p_slip": round(float((sel["sl"] > 0).mean()) * 100), "dur_mean": round(float(sel["du"].mean()), 2)})
        slip_r = {"rain": round(float(D["r"].corr(D["sl"])), 2), "dur": round(float(D["du"].corr(D["sl"])), 2), "n": int(len(D))}
    else:
        slip_r = None

    c = clim_of(S)
    out = {"generated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "target": target, "origin": str(last),
           "series": "gabungan %s–%s" % (S.index[0], last), "order": order, "methods": F, "notes": notes,
           "clim": [round(float(c[m]), 1) for m in range(1, 13)],
           "p50": F.get("Persentil P50 (median)"), "p80": F.get("Persentil P80"),
           "skill": skill, "weights": {k: round(v, 3) for k, v in w.items()},
           "enso_now": None if enso is None else round(float(enso.tail(3).mean()), 2),
           "iod_now": None if iod is None else round(float(iod.tail(3).mean()), 2),
           "freq": fr, "slip": slip, "slip_r": slip_r}
    if os.environ.get("JFEW_LOCAL_INPUT"):
        json.dump(out, open("rain_fc_out.json", "w"), indent=1)
        print(json.dumps({k: out[k] for k in ["target", "origin", "order", "weights"]}, indent=1))
        return
    import jfew_burn_job as J
    print(J.web({"action": "rain_fc", "result": out}))


if __name__ == "__main__":
    main()
