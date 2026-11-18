"""
J-FEW · HINDCAST RISIKO KEBAKARAN (jfew_risk_hindcast.py · v1 · 18 Nov 2026) — SEKALI JALAN, TIDAK MASUK SISTEM
Menghitung ulang peta "Risiko kebakaran" untuk SETIAP HARI dari HINDCAST_START (default 2026-07-01) s.d. kemarin
dengan model & ambang kelas yang sama dengan peta harian (ml_model/jfew_ml_model.joblib), lalu menulis
RATA-RATA PER BULAN luas (ha) kelas Sedang / Tinggi / Sangat tinggi ke halaman Summary GitHub Actions.
· Tidak mengirim apa pun ke Apps Script / spreadsheet / Telegram (tanpa JFEW_WEBAPP_URL).
· Cuaca: Open-Meteo archive (ERA5) + forecast past_days untuk hari terakhir yang belum ada di arsip — definisi sama dengan peta harian.
· Luas konsesi & reklamasi: jfew_hindcast_areas.json (batas konsesi PT JBG & reklamasi kumulatif 2005–Q2 2026).
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT · opsional HINDCAST_START (YYYY-MM-DD), HINDCAST_END, TANAM_RP (default 28000000), KURS (17800)
"""
import datetime as dt
import json
import os

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import box, shape

import jfew_burn_job as J
import jfew_ml_dataset as D
import jfew_fire_risk as R

WITA = dt.timezone(dt.timedelta(hours=8))
CLS = ["Rendah", "Sedang", "Tinggi", "Sangat tinggi"]
BLN = ["", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
TANAM = float(os.environ.get("TANAM_RP") or 28e6)
KURS = float(os.environ.get("KURS") or 17800)


def weather_hist(t0, t1):
    """per titik PTS: DataFrame harian rain, tmax, rh, wind (definisi = R.weather) untuk t0..t1"""
    lat = ",".join(str(p[1]) for p in R.PTS); lon = ",".join(str(p[0]) for p in R.PTS)
    hourly = "temperature_2m,dew_point_2m,wind_speed_10m,wind_direction_10m"
    arch = R.get_json("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s"
                      "&daily=precipitation_sum,temperature_2m_max&hourly=%s&wind_speed_unit=ms&timezone=GMT" % (lat, lon, t0, t1, hourly))
    arch = arch if isinstance(arch, list) else [arch]
    rain_l = R.get_json("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s"
                        "&daily=precipitation_sum&models=era5_land&timezone=GMT" % (lat, lon, t0, t1))
    rain_l = rain_l if isinstance(rain_l, list) else [rain_l]
    fc = R.get_json("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&daily=precipitation_sum,temperature_2m_max"
                    "&hourly=%s&wind_speed_unit=ms&timezone=GMT&past_days=92&forecast_days=1" % (lat, lon, hourly))
    fc = fc if isinstance(fc, list) else [fc]

    def frame(d):
        day = pd.DataFrame({"date": d["daily"]["time"], "rain": d["daily"]["precipitation_sum"], "tmax": d["daily"]["temperature_2m_max"]}).set_index("date")
        h = pd.DataFrame(d["hourly"]); h["date"] = h.time.str[:10]
        rad = np.deg2rad(h.wind_direction_10m)
        h["u"] = -h.wind_speed_10m * np.sin(rad); h["v"] = -h.wind_speed_10m * np.cos(rad)
        g = h.groupby("date").agg(t=("temperature_2m", "mean"), td=("dew_point_2m", "mean"), u=("u", "mean"), v=("v", "mean"))
        g["rh"] = 100 * np.exp(17.625 * g.td / (g.td + 243.04)) / np.exp(17.625 * g.t / (g.t + 243.04))
        g["wind"] = np.hypot(g.u, g.v)
        return day.join(g[["rh", "wind"]], how="left")

    out = []
    for i in range(len(R.PTS)):
        a = frame(arch[i]).dropna(how="all")
        er = pd.Series(rain_l[i]["daily"]["precipitation_sum"], index=rain_l[i]["daily"]["time"], dtype=float).dropna()
        er = er[er.index.isin(a.index)]
        a.loc[er.index, "rain"] = er.values                     # hujan ERA5-Land seperti peta harian
        f = frame(fc[i])
        miss = [x for x in f.index if x not in a.index or a.loc[x].isna().any()]
        a = pd.concat([a.drop(index=[x for x in miss if x in a.index]), f.loc[miss]]).sort_index()
        out.append(a)
    return out


def cell_fracs(cells, cell=0.009):
    g = json.load(open("jfew_hindcast_areas.json"))
    kon, rek = shape(g["konsesi"]).buffer(0), shape(g["reklamasi"]).buffer(0)
    h = cell / 2; res = {}
    for lo, la in cells:
        b = box(lo - h, la - h, lo + h, la + h); a = b.area
        ha = cell * 111.32 * np.cos(np.radians(la)) * cell * 110.574 * 100
        res[(lo, la)] = (ha, kon.intersection(b).area / a if kon.intersects(b) else 0.0, rek.intersection(b).area / a if rek.intersects(b) else 0.0)
    return res


def main():
    pk = joblib.load(R.MODEL)
    model, feats, thr = pk["model"], pk["features"], pk["thr"]
    J.init_ee(); D.init_layers()
    yesterday = dt.datetime.now(WITA).date() - dt.timedelta(1)
    d0 = dt.date.fromisoformat(os.environ.get("HINDCAST_START") or "2026-07-01")
    d1 = dt.date.fromisoformat(os.environ.get("HINDCAST_END") or yesterday.isoformat())
    wx = weather_hist((d0 - dt.timedelta(95)).isoformat(), d1.isoformat())
    regional = pd.concat([w.rain for w in wx], axis=1).mean(axis=1).fillna(0)
    nino = dmi = {}
    if "nino34" in feats or "dmi" in feats:
        nino = D.psl("https://psl.noaa.gov/data/correlation/nina34.anom.data")
        dmi = D.psl("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data")
    cells = D.grid()
    near = {c: min(range(len(R.PTS)), key=lambda i: (R.PTS[i][0] - c[0]) ** 2 + (R.PTS[i][1] - c[1]) ** 2) for c in cells}
    FR = cell_fracs(cells)
    rows, st, st_day = [], None, None
    t = d0
    while t <= d1:
        target = t.isoformat()
        if st is None or (t - st_day).days >= 8:                  # NDVI & lapisan statis diperbarui tiap 8 hari
            st, st_day = R.static_layers(cells, target), t
        wf = [R.weather_feats(w, target) for w in wx]
        cdd, dsr = R.runs(regional, target)
        common = {"cdd": cdd, "dsr5": dsr, "nino34": R.index_latest(nino, target) if nino else None,
                  "dmi": R.index_latest(dmi, target) if dmi else None, "month": t.month, "doy": t.timetuple().tm_yday}
        X, keys = [], []
        for c in cells:
            if c not in st or st[c].get("landcover") == 80:
                continue
            r = dict(st[c]); r.update(wf[near[c]]); r.update(common)
            if any(r.get(f) is None or (isinstance(r.get(f), float) and np.isnan(r.get(f))) for f in feats):
                continue
            X.append([r[f] for f in feats]); keys.append(c)
        acc = {"tot": [0.0] * 4, "kon": [0.0] * 4, "rek": [0.0] * 4, "n": [0] * 4}
        if X:
            score = model.predict_proba(pd.DataFrame(X, columns=feats))[:, 1]
            for c, s in zip(keys, score):
                k = 3 if s >= thr["p97"] else 2 if s >= thr["p90"] else 1 if s >= thr["p75"] else 0
                ha, fk, fr = FR[c]
                acc["n"][k] += 1; acc["tot"][k] += ha; acc["kon"][k] += ha * fk; acc["rek"][k] += ha * fr
        level = next((k for k in (3, 2, 1) if sum(acc["n"][k:]) >= 3), 0)
        rows.append({"date": target, "month": target[:7], "level": level, "cells": len(keys),
                     **{"%s%d" % (g, k): acc[g][k] for g in ("tot", "kon", "rek") for k in range(4)}})
        print("%s · tingkat %s · sel %s · reklamasi Tinggi+ %.0f ha" % (target, CLS[level], acc["n"], acc["rek"][2] + acc["rek"][3]))
        t += dt.timedelta(1)

    df = pd.DataFrame(rows)
    os.makedirs("hindcast_out", exist_ok=True); df.to_csv("hindcast_out/jfew_risk_hindcast_harian.csv", index=False)
    md = ["# J-FEW · Hindcast risiko kebakaran %s s.d. %s (TIDAK masuk sistem)" % (d0, d1), "",
          "Model %s (latih %s) · ambang: Sedang ≥ %.3f · Tinggi ≥ %.3f · Sangat tinggi ≥ %.3f · cuaca Open-Meteo archive (ERA5) · sel 1 km" %
          (pk.get("best"), pk.get("trained"), thr["p75"], thr["p90"], thr["p97"]), ""]
    for g, title in (("tot", "Seluruh grid"), ("kon", "Dalam konsesi"), ("rek", "Area reklamasi")):
        md += ["## Rata-rata luas per hari (ha) · %s" % title, "",
               "| Bulan | Hari | Sedang | Tinggi | Sangat tinggi | Tinggi + Sangat tinggi |", "|---|---|---|---|---|---|"]
        for m, x in df.groupby("month"):
            a = [x["%s%d" % (g, k)].mean() for k in range(4)]
            md.append("| %s %s | %d | %s | %s | %s | **%s** |" % (BLN[int(m[5:])], m[:4], len(x), *["{:,.0f}".format(v) for v in a[1:]], "{:,.0f}".format(a[2] + a[3])))
        md.append("")
    md += ["## Jumlah hari per tingkat area (≥3 sel)", "", "| Bulan | Rendah | Sedang | Tinggi | Sangat tinggi |", "|---|---|---|---|---|"]
    for m, x in df.groupby("month"):
        md.append("| %s %s | %s |" % (BLN[int(m[5:])], m[:4], " | ".join(str(int((x.level == k).sum())) for k in range(4))))
    md += ["", "## Nilai tanam ulang reklamasi yang berisiko (rata-rata per hari · Rp %s jt/ha · kurs Rp %s)" % ("{:,.0f}".format(TANAM / 1e6), "{:,.0f}".format(KURS)), "",
           "| Bulan | Sedang (Rp M) | Tinggi (Rp M) | Sangat tinggi (Rp M) | Tinggi + Sangat tinggi (ribu USD) |", "|---|---|---|---|---|"]
    for m, x in df.groupby("month"):
        a = [x["rek%d" % k].mean() * TANAM / 1e9 for k in range(4)]
        md.append("| %s %s | %.1f | %.1f | %.1f | %s |" % (BLN[int(m[5:])], m[:4], a[1], a[2], a[3], "{:,.0f}".format((a[2] + a[3]) * 1e9 / KURS / 1000)))
    md += ["", "Rata-rata = jumlah luas kelas tsb tiap hari ÷ jumlah hari di bulan itu. Rp M = miliar. Detail harian: artefak jfew_risk_hindcast_harian.csv."]
    txt = "\n".join(md)
    print(txt)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(txt)


if __name__ == "__main__":
    main()
