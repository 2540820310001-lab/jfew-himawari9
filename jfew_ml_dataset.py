#!/usr/bin/env python3
"""
J-FEW · MACHINE LEARNING TAHAP 1 — TABEL DATA PELUANG KEBAKARAN HARIAN (jfew_ml_dataset.py · v1 · 11 Nov 2026)
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


ERA = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
NDVI = ee.ImageCollection("MODIS/061/MOD13A1").select("NDVI")
FIRMS = ee.ImageCollection("FIRMS").select("T21")
WC = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")
DEM = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30)
STATIC = WC.addBands(DEM.rename("elev")).addBands(ee.Terrain.slope(DEM).rename("slope"))


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
    hist = FIRMS.filterDate("2000-11-01", ee.Date.fromYMD(d.get("year"), 1, 1)).map(lambda i: i.gt(0).unmask(0).toByte()).merge(zero).sum().rename("fire_hist")
    lab = FIRMS.filterDate(d, d.advance(1, "day")).map(lambda i: i.gt(0).unmask(0).toByte()).merge(zero).max().rename("label")
    return (tmax.addBands(rh).addBands(wind).addBands(rain).addBands(prev(7).rename("rain7")).addBands(prev(30).rename("rain30"))
            .addBands(prev(90).rename("rain90")).addBands(dry).addBands(ndvi).addBands(hist).addBands(STATIC).addBands(lab))


BANDS = ["tmax", "rh", "wind", "rain", "rain7", "rain30", "rain90", "dry30", "ndvi", "fire_hist", "landcover", "elev", "slope", "label"]


def sample_dates(plan):
    """plan: list (date, [(lon,lat),...]) → baris fitur"""
    rows = []
    for i in range(0, len(plan), 20):
        chunk = plan[i:i + 20]
        def one(item):
            date, pts = item
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"lon": lo, "lat": la, "date": date}) for lo, la in pts])
            return stack(date).sampleRegions(collection=fc, scale=SCALE, geometries=False)
        fcs = ee.FeatureCollection([one(it) for it in chunk]).flatten()
        res = retry(lambda: fcs.getInfo())
        for f in res["features"]:
            p = f["properties"]
            if any(p.get(b) is None for b in BANDS):
                continue
            rows.append(p)
        if (i // 20) % 10 == 0:
            print("  tanggal %d/%d · baris %d" % (min(i + 20, len(plan)), len(plan), len(rows)))
    return rows


def main():
    J.init_ee()
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
    cols = ["date", "lon", "lat"] + BANDS
    with open(OUT + "/jfew_ml_dataset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["month", "doy", "in_core"], extrasaction="ignore"); w.writeheader()
        for r in rows:
            d = dt.date.fromisoformat(r["date"])
            r["month"], r["doy"] = d.month, d.timetuple().tm_yday
            r["in_core"] = int(CORE[0] <= r["lon"] <= CORE[2] and CORE[1] <= r["lat"] <= CORE[3])
            w.writerow(r)
    pos = sum(1 for r in rows if r["label"] >= 0.5)
    # --- uji awal (indikatif)
    auc_txt = ""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, average_precision_score
        df = pd.read_csv(OUT + "/jfew_ml_dataset.csv")
        df["label"] = (df["label"] >= 0.5).astype(int)
        df["year"] = df["date"].str[:4].astype(int)
        feats = ["tmax", "rh", "wind", "rain", "rain7", "rain30", "rain90", "dry30", "ndvi", "fire_hist", "landcover", "elev", "slope", "month", "doy"]
        tr, te = df[df.year <= 2018], df[df.year >= 2019]
        if tr.label.sum() > 20 and te.label.sum() > 20:
            rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=1).fit(tr[feats], tr.label)
            p = rf.predict_proba(te[feats])[:, 1]
            base = LogisticRegression(max_iter=1000).fit(tr[["rain30"]], tr.label).predict_proba(te[["rain30"]])[:, 1]
            imp = sorted(zip(feats, rf.feature_importances_), key=lambda x: -x[1])[:6]
            auc_txt = ("| Model | AUC | Average precision |\n|---|---|---|\n| Random Forest (semua fitur) | %.3f | %.3f |\n| Sederhana (hujan 30 hari saja) | %.3f | %.3f |\n\n"
                       "Fitur paling berpengaruh: %s\n" % (roc_auc_score(te.label, p), average_precision_score(te.label, p),
                                                          roc_auc_score(te.label, base), average_precision_score(te.label, base),
                                                          ", ".join("%s (%.2f)" % (k, v) for k, v in imp)))
    except Exception as ex:                                      # noqa: BLE001
        auc_txt = "Uji awal dilewati: %s\n" % ex
    md = ["# J-FEW ML tahap 1 — tabel data peluang kebakaran", "",
          "Wilayah %s–%s BT, %s–%s LS · %d–%d · sumber label MODIS FIRMS (1 km)" % (W, E, -N, -S, START, END), "",
          "**Piksel-hari ber-api:** %d (di amplop konsesi + buffer: %d) pada **%d hari**" % (len(fires), core, len(fire_dates)), "",
          "| Tahun | Piksel-hari api |", "|---|---|"] + ["| %s | %d |" % (y, by_year.get(str(y), 0)) for y in range(START, END + 1)] + \
         ["", "| Bulan | Piksel-hari api |", "|---|---|"] + ["| %d | %d |" % (m, by_month.get(m, 0)) for m in range(1, 13)] + \
         ["", "**Tabel data:** %d baris, %d berlabel api (1), %d tanpa api (0)" % (len(rows), pos, len(rows) - pos), "",
          "## Uji awal (latih ≤2018, uji 2019–%d) — indikatif" % END, "", auc_txt]
    open(OUT + "/jfew_ml_summary.md", "w").write("\n".join(md))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
