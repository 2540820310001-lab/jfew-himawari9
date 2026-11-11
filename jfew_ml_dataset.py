#!/usr/bin/env python3
"""
J-FEW · MACHINE LEARNING TAHAP 1 — TABEL DATA PELUANG KEBAKARAN HARIAN (jfew_ml_dataset.py · v2 · 11 Nov 2026 · + ENSO/IOD, kekeringan beruntun, tutupan lahan tahunan, jarak permukiman, RF vs gradient boosting, peluang terkalibrasi)
------------------------------------------------------------------------------------------------------------
Pertanyaan model: "Hari ini/besok, sel 1 km mana di sekitar PT JBG yang paling mungkin muncul titik api?"
Wilayah   : kotak 114.92–115.14 BT, 3.82–3.96 LS (konsesi + buffer + ±3–4 km sekitarnya), sel ±1 km
Periode   : 2001–tahun lalu (arsip MODIS FIRMS di Google Earth Engine, sensor konsisten sejak Nov 2000)
Label     : 1 = ada piksel api MODIS (FIRMS, T21) di sel itu pada hari itu; 0 = tidak ada
Sampel    : semua sel-hari yang ada api + 30 sel acak per tanggal; tanggal = semua hari ber-api
            + hari tanpa api acak (2× jumlah hari ber-api) → seimbang antar musim
Fitur (tersedia pada saat prakiraan dibuat):
  cuaca hari itu (ERA5-Land): suhu maks, RH, angin, hujan · hujan 7/30/90 hari sebelumnya · hari kering (<1 mm) 30 hari
  vegetasi: NDVI MODIS 16-harian terakhir sebelum tanggal · tutupan lahan ESA WorldCover · elevasi & lereng (Copernicus DEM)
  riwayat: jumlah hari ber-api di sel tsb pada tahun-tahun sebelumnya · bulan, hari ke-n
Keluaran (artefak GitHub Actions): jfew_ml_dataset.csv, jfew_ml_fire_days.csv, ringkasan di halaman Summary
Uji awal (indikatif): Random Forest dilatih ≤2018, diuji 2019–tahun lalu → AUC vs model sederhana (hujan 30 hari saja).
v2 menambah: Niño3.4 & DMI (NOAA PSL, rata-rata 3 bulan terakhir yang tersedia) · hari kering beruntun & hari sejak hujan ≥5 mm
  (seri hujan harian ERA5-Land regional) · tutupan lahan MODIS MCD12Q1 per tahun · jarak ke permukiman (GHSL) ·
  perbandingan Random Forest vs HistGradientBoosting, kalibrasi isotonik (latih ≤2014, kalibrasi 2015–2018, uji 2019+),
  model akhir dilatih ulang 2001–tahun lalu → artefak jfew_ml_model.joblib (dipakai prakiraan harian tahap 3).
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT · opsional ML_START (default 2001), ML_END (default tahun lalu), ML_NEG_PER_DAY (30)
"""
import csv
import datetime as dt
import os
import random
import time

import ee

import jfew_burn_job as J

W, S, E, N = 114.92, -3.96, 115.14, -3.82
CORE = (114.9546, -3.9286, 115.1042, -3.8588)            # amplop konsesi + buffer 1 km J-FEW
START = int(os.environ.get("ML_START") or 2001)
END = int(os.environ.get("ML_END") or dt.date.today().year - 1)
NEG_PER_DAY = int(os.environ.get("ML_NEG_PER_DAY") or 30)
SCALE = 1000
random.seed(42)
OUT = "ml_out"


def retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn()
        except Exception as ex:                                  # noqa: BLE001
            if k == tries - 1:
                raise
            print("  ulang:", str(ex)[:160]); time.sleep(15 * (k + 1))


def fire_days(aoi):
    """daftar hari ber-api + koordinat piksel api MODIS (FIRMS T21) di dalam kotak"""
    out = []
    for y in range(START, END + 1):
        col = ee.ImageCollection("FIRMS").filterDate("%d-01-01" % y, "%d-01-01" % (y + 1)).select("T21")
        def per_img(img):
            pts = img.gt(0).selfMask().rename("f").sample(region=aoi, scale=SCALE, geometries=True, dropNulls=True)
            d = img.date().format("YYYY-MM-dd")
            return pts.map(lambda f: f.set("date", d))
        fc = retry(lambda: col.map(per_img).flatten().getInfo())
        for f in fc["features"]:
            lon, lat = f["geometry"]["coordinates"]
            out.append((f["properties"]["date"], round(lon, 4), round(lat, 4)))
        print(y, "piksel-hari api:", sum(1 for o in out if o[0].startswith(str(y))))
    return out


def grid():
    pts, lat = [], S + 0.0045
    while lat < N:
        lon = W + 0.0045
        while lon < E:
            pts.append((round(lon, 4), round(lat, 4))); lon += 0.009
        lat += 0.009
    return pts


ERA = NDVI = FIRMS = STATIC = YEARLY = MCD = None
OPT_BANDS = []                                              # pita opsional yang berhasil dimuat (lc_year, dist_built)


def init_layers():
    """dibuat SETELAH ee.Initialize (objek Earth Engine tidak boleh dibuat sebelum login)"""
    global ERA, NDVI, FIRMS, STATIC, YEARLY, MCD
    ERA = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    NDVI = ee.ImageCollection("MODIS/061/MOD13A1").select("NDVI")
    FIRMS = ee.ImageCollection("FIRMS").select("T21")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30)
    STATIC = wc.addBands(dem.rename("elev")).addBands(ee.Terrain.slope(dem).rename("slope"))
    try:                                                    # jarak ke permukiman (GHSL built-up 2020, >5 % terbangun per 100 m)
        built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").gt(500)
        dist = built.fastDistanceTransform(40).sqrt().multiply(ee.Image.pixelArea().sqrt()).divide(1000).rename("dist_built")
        dist.bandNames().getInfo()
        STATIC = STATIC.addBands(dist); OPT_BANDS.append("dist_built")
    except Exception as ex:                                 # noqa: BLE001
        print("GHSL dilewati:", str(ex)[:120])
    try:
        MCD = ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1")
        MCD.size().getInfo(); OPT_BANDS.append("lc_year")
    except Exception as ex:                                 # noqa: BLE001
        MCD = None; print("MCD12Q1 dilewati:", str(ex)[:120])
    # ringkasan tahunan (hari ber-api per piksel per tahun) — jauh lebih ringan daripada menjumlah ribuan citra harian tiap tanggal
    YEARLY = ee.ImageCollection([FIRMS.filterDate("%d-01-01" % y, "%d-01-01" % (y + 1)).map(lambda i: i.gt(0).unmask(0).toByte())
                                 .sum().toInt16().rename("T21").set("year", y) for y in range(2001, END + 1)])


def stack(date):
    d = ee.Date(date)
    day = ERA.filterDate(d, d.advance(1, "day")).first()
    t = ee.Image(day).select("temperature_2m").subtract(273.15)
    td = ee.Image(day).select("dewpoint_temperature_2m").subtract(273.15)
    rh = td.multiply(17.625).divide(td.add(243.04)).exp().divide(t.multiply(17.625).divide(t.add(243.04)).exp()).multiply(100).rename("rh")
    wind = ee.Image(day).select("u_component_of_wind_10m").hypot(ee.Image(day).select("v_component_of_wind_10m")).rename("wind")
    tmax = ee.Image(day).select("temperature_2m_max").subtract(273.15).rename("tmax")
    rain = ee.Image(day).select("total_precipitation_sum").multiply(1000).rename("rain")
    def prev(n):
        return ERA.filterDate(d.advance(-n, "day"), d).select("total_precipitation_sum").sum().multiply(1000)
    dry = ERA.filterDate(d.advance(-30, "day"), d).select("total_precipitation_sum").map(lambda i: i.lt(0.001)).sum().rename("dry30")
    nd = NDVI.filterDate(d.advance(-40, "day"), d).sort("system:time_start", False).first()
    ndvi = ee.Image(ee.Algorithms.If(nd, ee.Image(nd).multiply(0.0001), ee.Image.constant(-1))).rename("ndvi")
    zero = ee.ImageCollection([ee.Image.constant(0).toByte().rename("T21")])   # cegah koleksi kosong (hari tanpa citra FIRMS)
    hist = YEARLY.filter(ee.Filter.lt("year", d.get("year"))).merge(zero).sum().rename("fire_hist")   # jumlah hari ber-api tahun-tahun sebelumnya
    lab = FIRMS.filterDate(d, d.advance(1, "day")).map(lambda i: i.gt(0).unmask(0).toByte()).merge(zero).max().rename("label")
    extra = ee.Image([])
    if MCD is not None:
        lcy = MCD.filter(ee.Filter.calendarRange(d.get("year"), d.get("year"), "year")).first()
        lcy = ee.Image(ee.Algorithms.If(lcy, lcy, MCD.sort("system:time_start", False).first()))
        extra = extra.addBands(lcy.rename("lc_year"))
    return (extra.addBands(tmax).addBands(rh).addBands(wind).addBands(rain).addBands(prev(7).rename("rain7")).addBands(prev(30).rename("rain30"))
            .addBands(prev(90).rename("rain90")).addBands(dry).addBands(ndvi).addBands(hist).addBands(STATIC).addBands(lab))


BANDS = ["tmax", "rh", "wind", "rain", "rain7", "rain30", "rain90", "dry30", "ndvi", "fire_hist", "landcover", "elev", "slope", "label"]


def sample_dates(plan):
    """plan: list (date, [(lon,lat),...]) → baris fitur"""
    rows = []
    for i in range(0, len(plan), 10):
        chunk = plan[i:i + 10]
        def one(item):
            date, pts = item
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"lon": lo, "lat": la, "date": date}) for lo, la in pts])
            return stack(date).sampleRegions(collection=fc, scale=SCALE, geometries=False)
        fcs = ee.FeatureCollection([one(it) for it in chunk]).flatten()
        res = retry(lambda: fcs.getInfo())
        for f in res["features"]:
            p = f["properties"]
            if any(p.get(b) is None for b in BANDS + OPT_BANDS):
                continue
            rows.append(p)
        if (i // 10) % 10 == 0:
            print("  tanggal %d/%d · baris %d" % (min(i + 10, len(plan)), len(plan), len(rows)), flush=True)
    return rows


def regional_rain():
    """hujan harian ERA5-Land rata-rata kotak → hari kering beruntun & hari sejak hujan ≥5 mm (per tanggal)"""
    import datetime as _dt
    aoi = ee.Geometry.Rectangle([W, S, E, N]); ser = {}
    for y in range(START - 1, END + 1):
        col = ERA.filterDate("%d-01-01" % y, "%d-01-01" % (y + 1)).select("total_precipitation_sum")
        fc = col.map(lambda i: ee.Feature(None, {"d": i.date().format("YYYY-MM-dd"),
                                                  "p": i.reduceRegion(ee.Reducer.mean(), aoi, 11132).get("total_precipitation_sum")}))
        for f in retry(lambda: fc.getInfo())["features"]:
            v = f["properties"].get("p")
            if v is not None:
                ser[f["properties"]["d"]] = v * 1000
    out, cdd, dsr = {}, 0, 0
    for d in sorted(ser):
        out[d] = (cdd, dsr)                                 # nilai s.d. kemarin (tersedia saat prakiraan)
        cdd = cdd + 1 if ser[d] < 1 else 0
        dsr = dsr + 1 if ser[d] < 5 else 0
    print("seri hujan regional:", len(ser), "hari")
    return out


def psl(url):
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "J-FEW ML"}), timeout=60).read().decode()
    s = {}
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) == 13 and p[0].isdigit() and 1850 < int(p[0]) < 2200:
            for m, v in enumerate(p[1:], 1):
                try:
                    x = float(v)
                except ValueError:
                    continue
                if x > -90:
                    s[(int(p[0]), m)] = x
    return s


def climate_index(s, date):
    """rata-rata 3 bulan sebelum bulan tanggal tsb (sudah terbit saat prakiraan)"""
    y, m = int(date[:4]), int(date[5:7]); v = []
    for k in range(1, 4):
        mm, yy = m - k, y
        while mm <= 0:
            mm += 12; yy -= 1
        if (yy, mm) in s:
            v.append(s[(yy, mm)])
    return sum(v) / len(v) if v else None


def model_report(path):
    import json
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    df = pd.read_csv(path)
    df["label"] = (df["label"] >= 0.5).astype(int)
    df["year"] = df["date"].str[:4].astype(int)
    feats = [c for c in ["tmax", "rh", "wind", "rain", "rain7", "rain30", "rain90", "dry30", "cdd", "dsr5", "nino34", "dmi", "ndvi",
                         "fire_hist", "landcover", "lc_year", "elev", "slope", "dist_built", "month", "doy"] if c in df.columns]
    df = df.dropna(subset=feats)
    tr, ca, te = df[df.year <= 2014], df[(df.year >= 2015) & (df.year <= 2018)], df[df.year >= 2019]
    def w(y):
        pos = y.mean(); return np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    models = {"Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=1),
              "Gradient boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=1)}
    res, fitted = [], {}
    for name, m in models.items():
        if name == "Gradient boosting":
            m.fit(tr[feats], tr.label, sample_weight=w(tr.label))
        else:
            m.fit(tr[feats], tr.label)
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(ca[feats])[:, 1], ca.label)
        p = iso.predict(m.predict_proba(te[feats])[:, 1])
        k = max(1, int(0.1 * len(te))); top = np.argsort(-p)[:k]
        res.append([name, roc_auc_score(te.label, p), average_precision_score(te.label, p), brier_score_loss(te.label, p),
                    te.label.values[top].sum() / te.label.sum()])
        fitted[name] = (m, iso)
    base = LogisticRegression(max_iter=1000).fit(tr[["rain30"]], tr.label)
    pb = base.predict_proba(te[["rain30"]])[:, 1]; k = max(1, int(0.1 * len(te))); top = np.argsort(-pb)[:k]
    res.append(["Sederhana (hujan 30 hari)", roc_auc_score(te.label, pb), average_precision_score(te.label, pb), brier_score_loss(te.label, pb),
                te.label.values[top].sum() / te.label.sum()])
    best = max(res[:2], key=lambda r: r[2])[0]
    per_year = []
    m, iso = fitted[best]
    for y in sorted(te.year.unique()):
        t = te[te.year == y]
        if t.label.sum() >= 5:
            per_year.append((y, int(t.label.sum()), average_precision_score(t.label, iso.predict(m.predict_proba(t[feats])[:, 1]))))
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(m, te[feats], te.label, scoring="average_precision", n_repeats=5, random_state=1, n_jobs=-1)
    imp = sorted(zip(feats, pi.importances_mean), key=lambda x: -x[1])[:8]
    # model akhir: latih ulang semua tahun (≤ tahun lalu), kalibrasi pada 2015+ dengan model ≤2014 tidak dipakai → isotonic dari validasi silang waktu sederhana
    final = models[best].__class__(**models[best].get_params())
    if best == "Gradient boosting":
        final.fit(df[feats], df.label, sample_weight=w(df.label))
    else:
        final.fit(df[feats], df.label)
    prev_sample = float(df.label.mean())
    joblib.dump({"model": final, "iso": iso, "features": feats, "best": best, "prev_sample": prev_sample,
                 "trained": "2001-%d" % END, "note": "peluang = relatif thd sampel; koreksi prevalensi di tahap 3"}, OUT + "/jfew_ml_model.joblib")
    json.dump({"results": res, "per_year": per_year, "importance": imp, "best": best, "features": feats}, open(OUT + "/jfew_ml_metrics.json", "w"), default=float, indent=1)
    t = ["| Model | AUC | Average precision | Brier | % api tertangkap di 10 % sel teratas |", "|---|---|---|---|---|"]
    t += ["| %s%s | %.3f | %.3f | %.3f | %.0f%% |" % (r[0], " ✓" if r[0] == best else "", r[1], r[2], r[3], r[4] * 100) for r in res]
    t += ["", "Per tahun (model terpilih): " + " · ".join("%d: AP %.2f (%d api)" % (y, ap, n) for y, n, ap in per_year), "",
          "Fitur paling berpengaruh (permutation importance, Δ AP): " + ", ".join("%s (%.3f)" % (k, v) for k, v in imp), "",
          "Fitur dipakai (%d): %s" % (len(feats), ", ".join(feats))]
    return "\n".join(t)


def main():
    J.init_ee()
    init_layers()
    os.makedirs(OUT, exist_ok=True)
    aoi = ee.Geometry.Rectangle([W, S, E, N])
    fires = fire_days(aoi)
    with open(OUT + "/jfew_ml_fire_days.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["date", "lon", "lat", "in_core"])
        for d, lo, la in fires:
            w.writerow([d, lo, la, int(CORE[0] <= lo <= CORE[2] and CORE[1] <= la <= CORE[3])])
    # --- ringkasan kejadian historis
    by_year, by_month, days = {}, {}, {}
    for d, lo, la in fires:
        by_year[d[:4]] = by_year.get(d[:4], 0) + 1
        by_month[int(d[5:7])] = by_month.get(int(d[5:7]), 0) + 1
        days.setdefault(d, []).append((lo, la))
    core = sum(1 for d, lo, la in fires if CORE[0] <= lo <= CORE[2] and CORE[1] <= la <= CORE[3])
    # --- rencana sampel
    G = grid()
    fire_dates = sorted(days)
    all_days = [dt.date(START, 1, 1) + dt.timedelta(k) for k in range((dt.date(END, 12, 31) - dt.date(START, 1, 1)).days + 1)]
    pool = [d.isoformat() for d in all_days if d.isoformat() not in days]
    neg_dates = random.sample(pool, min(len(pool), 2 * len(fire_dates)))
    plan = []
    for d in fire_dates + neg_dates:
        plan.append((d, days.get(d, []) + random.sample(G, NEG_PER_DAY)))
    plan.sort()
    print("rencana: %d tanggal (%d ber-api), %d titik" % (len(plan), len(fire_dates), sum(len(p[1]) for p in plan)))
    rows = sample_dates(plan)
    reg = regional_rain()
    try:
        nino = psl("https://psl.noaa.gov/data/correlation/nina34.anom.data")
    except Exception as ex:                                 # noqa: BLE001
        nino = {}; print("Niño3.4 gagal:", ex)
    try:
        dmi = psl("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data")
    except Exception as ex:                                 # noqa: BLE001
        dmi = {}; print("DMI gagal:", ex)
    for r in rows:
        r["cdd"], r["dsr5"] = reg.get(r["date"], (None, None))
        r["nino34"], r["dmi"] = climate_index(nino, r["date"]), climate_index(dmi, r["date"])
    cols = ["date", "lon", "lat"] + BANDS + OPT_BANDS + ["cdd", "dsr5", "nino34", "dmi"]
    with open(OUT + "/jfew_ml_dataset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["month", "doy", "in_core"], extrasaction="ignore"); w.writeheader()
        for r in rows:
            d = dt.date.fromisoformat(r["date"])
            r["month"], r["doy"] = d.month, d.timetuple().tm_yday
            r["in_core"] = int(CORE[0] <= r["lon"] <= CORE[2] and CORE[1] <= r["lat"] <= CORE[3])
            w.writerow(r)
    pos = sum(1 for r in rows if r["label"] >= 0.5)
    # --- uji model (tahap 2)
    try:
        auc_txt = model_report(OUT + "/jfew_ml_dataset.csv")
    except Exception as ex:                                      # noqa: BLE001
        import traceback; traceback.print_exc()
        auc_txt = "Uji model gagal: %s\n" % ex
    md = ["# J-FEW ML tahap 2 — model peluang kebakaran", "",
          "Wilayah %s–%s BT, %s–%s LS · %d–%d · sumber label MODIS FIRMS (1 km)" % (W, E, -N, -S, START, END), "",
          "**Piksel-hari ber-api:** %d (di amplop konsesi + buffer: %d) pada **%d hari**" % (len(fires), core, len(fire_dates)), "",
          "| Tahun | Piksel-hari api |", "|---|---|"] + ["| %s | %d |" % (y, by_year.get(str(y), 0)) for y in range(START, END + 1)] + \
         ["", "| Bulan | Piksel-hari api |", "|---|---|"] + ["| %d | %d |" % (m, by_month.get(m, 0)) for m in range(1, 13)] + \
         ["", "**Tabel data:** %d baris, %d berlabel api (1), %d tanpa api (0)" % (len(rows), pos, len(rows) - pos), "",
          "## Uji model (latih ≤2014 · kalibrasi 2015–2018 · uji 2019–%d)" % END, "", auc_txt]
    open(OUT + "/jfew_ml_summary.md", "w").write("\n".join(md))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
