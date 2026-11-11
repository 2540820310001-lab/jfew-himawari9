"""
J-FEW · PETA RISIKO KEBAKARAN BESOK (jfew_fire_risk.py · v1.1 · 11 Nov 2026 · indeks iklim pakai 3 bulan terakhir yang tersedia · diagnosa fitur kosong)
Tiap pagi: cuaca Open-Meteo (92 hari lalu + prakiraan) + lapisan Earth Engine (NDVI, riwayat api, tutupan lahan, DEM,
jarak permukiman) + Niño3.4/DMI → model Random Forest (ml_model/jfew_ml_model.joblib, hasil jfew_ml_dataset.py)
→ skor per sel 1 km → 4 kelas (Rendah / Sedang / Tinggi / Sangat tinggi; ambang = persentil skor hari tanpa api)
→ dikirim ke Apps Script (action fire_risk) → layer "Risiko kebakaran besok" di peta. Tidak dikirim ke Telegram.
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT, JFEW_WEBAPP_URL, JFEW_BURN_TOKEN · opsional RISK_DAYS (default 2: besok & lusa)
"""
import datetime as dt
import json
import os
import urllib.request

import ee
import joblib
import numpy as np
import pandas as pd

import jfew_burn_job as J
import jfew_ml_dataset as D

MODEL = "ml_model/jfew_ml_model.joblib"
WITA = dt.timezone(dt.timedelta(hours=8))
PTS = [(lo, la) for la in (-3.86, -3.92) for lo in (114.95, 115.03, 115.11)]   # titik cuaca Open-Meteo (sel → titik terdekat)
CLS = ["Rendah", "Sedang", "Tinggi", "Sangat tinggi"]


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "J-FEW fire risk"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read().decode())


def weather():
    """per titik: DataFrame harian (tanggal WITA) rain, tmax, rh, wind — 92 hari lalu s.d. prakiraan +3 hari"""
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&daily=precipitation_sum,temperature_2m_max"
           "&hourly=relative_humidity_2m,wind_speed_10m&wind_speed_unit=ms&timezone=Asia%%2FMakassar&past_days=92&forecast_days=4"
           % (",".join(str(p[1]) for p in PTS), ",".join(str(p[0]) for p in PTS)))
    data = get_json(url)
    data = data if isinstance(data, list) else [data]
    out = []
    for d in data:
        day = pd.DataFrame({"date": d["daily"]["time"], "rain": d["daily"]["precipitation_sum"], "tmax": d["daily"]["temperature_2m_max"]})
        hr = pd.DataFrame({"t": d["hourly"]["time"], "rh": d["hourly"]["relative_humidity_2m"], "wind": d["hourly"]["wind_speed_10m"]})
        hr["date"] = hr.t.str[:10]
        day = day.merge(hr.groupby("date")[["rh", "wind"]].mean().reset_index(), on="date", how="left").set_index("date")
        out.append(day)
    return out


def weather_feats(day, target):
    """fitur cuaca untuk tanggal target (sama dengan definisi di jfew_ml_dataset.py)"""
    t = dt.date.fromisoformat(target)
    past = lambda n: [(t - dt.timedelta(k)).isoformat() for k in range(1, n + 1)]
    r = day.rain.fillna(0)
    f = {"tmax": day.tmax.get(target), "rh": day.rh.get(target), "wind": day.wind.get(target), "rain": r.get(target)}
    for n in (7, 30, 90):
        f["rain%d" % n] = float(sum(r.get(x, 0) for x in past(n)))
    f["dry30"] = int(sum(1 for x in past(30) if r.get(x, 0) < 1))
    return f


def runs(series, target):
    """hari kering beruntun (<1 mm) & hari sejak hujan ≥5 mm s.d. sehari sebelum target (rata-rata titik)"""
    cdd = dsr = 0
    for d in sorted(series.index):
        if d >= target:
            break
        cdd = cdd + 1 if series[d] < 1 else 0
        dsr = dsr + 1 if series[d] < 5 else 0
    return cdd, dsr


def static_layers(cells, target):
    """NDVI terbaru, riwayat api, tutupan lahan, DEM, jarak permukiman — sampel di pusat sel"""
    d = ee.Date(target)
    nd = D.NDVI.filterDate(d.advance(-90, "day"), d).sort("system:time_start", False).first()
    ndvi = ee.Image(nd).multiply(0.0001).rename("ndvi")
    hist = D.YEARLY.filter(ee.Filter.lt("year", d.get("year"))).sum().rename("fire_hist")
    img = ndvi.addBands(hist).addBands(D.STATIC)
    if D.MCD is not None:
        img = img.addBands(D.MCD.sort("system:time_start", False).first().rename("lc_year"))
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"lon": lo, "lat": la}) for lo, la in cells])
    res = D.retry(lambda: img.sampleRegions(collection=fc, scale=D.SCALE, geometries=False).getInfo())
    return {(f["properties"]["lon"], f["properties"]["lat"]): f["properties"] for f in res["features"]}


def index_latest(s, date):
    """rata-rata 3 bulan sebelum tanggal; bila belum terbit (DMI/Niño sering terlambat) → 3 bulan terakhir yang tersedia"""
    v = D.climate_index(s, date)
    if v is not None or not s:
        return v
    y, m = int(date[:4]), int(date[5:7])
    ks = sorted(k for k in s if k < (y, m))[-3:]
    return sum(s[k] for k in ks) / len(ks) if ks else None


def main():
    pk = joblib.load(MODEL)
    model, feats, thr = pk["model"], pk["features"], pk["thr"]
    J.init_ee()
    D.init_layers()
    today = dt.datetime.now(WITA).date()
    targets = [(today + dt.timedelta(k)).isoformat() for k in range(1, int(os.environ.get("RISK_DAYS") or 2) + 1)]
    wx = weather()
    regional = pd.concat([w.rain for w in wx], axis=1).mean(axis=1).fillna(0)
    try:
        nino = D.psl("https://psl.noaa.gov/data/correlation/nina34.anom.data")
        dmi = D.psl("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data")
    except Exception as ex:                                  # noqa: BLE001
        print("indeks iklim gagal:", ex); nino, dmi = {}, {}
    cells = D.grid()
    near = {c: min(range(len(PTS)), key=lambda i: (PTS[i][0] - c[0]) ** 2 + (PTS[i][1] - c[1]) ** 2) for c in cells}
    days, diag = [], []
    for target in targets:
        st = static_layers(cells, target)
        wf = [weather_feats(w, target) for w in wx]
        cdd, dsr = runs(regional, target)
        t = dt.date.fromisoformat(target)
        common = {"cdd": cdd, "dsr5": dsr, "nino34": index_latest(nino, target), "dmi": index_latest(dmi, target),
                  "month": t.month, "doy": t.timetuple().tm_yday}
        rows, keys, miss = [], [], {}
        print("%s · sel di grid %d · sel dengan data EE %d · cuaca titik-1 %s · iklim %s" % (target, len(cells), len(st), wf[0], common))
        for c in cells:
            if c not in st:
                miss["(data EE kosong)"] = miss.get("(data EE kosong)", 0) + 1; continue
            r = dict(st[c]); r.update(wf[near[c]]); r.update(common)
            bad = [f for f in feats if r.get(f) is None or (isinstance(r.get(f), float) and np.isnan(r.get(f)))]
            if bad:
                for f in bad:
                    miss[f] = miss.get(f, 0) + 1
                continue
            rows.append([r[f] for f in feats]); keys.append((c, r))
        if miss:
            print("  fitur kosong:", miss)
        if not rows:
            print(target, ": tidak ada sel lengkap"); diag.append("%s: tidak ada sel lengkap · fitur kosong %s" % (target, miss)); continue
        score = model.predict_proba(pd.DataFrame(rows, columns=feats))[:, 1]
        out = []
        for (c, r), s in zip(keys, score):
            k = 3 if s >= thr["p97"] else 2 if s >= thr["p90"] else 1 if s >= thr["p75"] else 0
            out.append([c[0], c[1], k, round(float(s), 3), round(r["rain30"]), round(r["rh"]), r["dsr5"], round(r["wind"], 1), round(r["tmax"], 1)])
        n = [sum(1 for o in out if o[2] == k) for k in range(4)]
        level = next((k for k in (3, 2, 1) if sum(n[k:]) >= 3), 0)
        days.append({"date": target, "level": level, "n": n, "cells": out,
                     "area": {"rain30": round(wf[0]["rain30"]), "dsr5": dsr, "cdd": cdd,
                              "nino34": common["nino34"], "dmi": common["dmi"]}})
        print("%s · tingkat area %s · sel per kelas %s" % (target, CLS[level], n))
    result = {"issued": dt.datetime.now(WITA).strftime("%Y-%m-%d %H:%M"), "cell": 0.009, "thr": thr, "model": pk.get("best"),
              "trained": pk.get("trained"), "days": days}
    print(J.web({"action": "fire_risk", "result": result}))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        md = ["# J-FEW · risiko kebakaran", "", "| Tanggal | Tingkat area | Rendah | Sedang | Tinggi | Sangat tinggi |", "|---|---|---|---|---|---|"]
        md += ["| %s | %s | %d | %d | %d | %d |" % (d["date"], CLS[d["level"]], *d["n"]) for d in days]
        md += ["", "Fitur dipakai model: " + ", ".join(feats)] + ["\n⚠ " + x for x in diag]
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))


if __name__ == "__main__":
    main()
