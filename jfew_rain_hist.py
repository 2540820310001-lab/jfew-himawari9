#!/usr/bin/env python3
"""
J-FEW · CURAH HUJAN SATELIT BULANAN di titik ombrometer PT JBG — jfew_rain_hist.py (v1.1 · 10 Nov 2026 · status Final/sementara dari metadata IMERG)
-----------------------------------------------------------------------------------------------------
Titik  : ombrometer PT JBG  UTM 50S 279948.364 E / 9568309.068 N  →  115.018279 E, -3.903238 (WGS84)
Periode: Jan 1998 – bulan terakhir yang tersedia (tiap produk punya tanggal akhir sendiri)
Produk (semua dijumlah per bulan, mm/bulan, nilai piksel tempat ombrometer berada):
  IMERG   NASA GPM IMERG V07 Final (0,1°, gauge-adjusted GPCC bulanan)   NASA/GPM_L3/IMERG_MONTHLY_V07
          + IMERG half-hourly (Late/Early) untuk bulan yang belum ada di produk bulanan → ditandai "prov"
  GSMAP   JAXA GSMaP v8 gauge-adjusted (hourlyPrecipRateGC, 0,1°, koreksi gauge harian NOAA/CPC)
  CHIRPS  CHIRPS v2.0 pentad (0,05°, IR + stasiun)                        UCSB-CHG/CHIRPS/PENTAD
  ERA5L   ERA5-Land reanalysis (0,1°, model, bukan satelit — pembanding)  ECMWF/ERA5_LAND/MONTHLY_AGGR
  PERSIANN PERSIANN-CDR (0,25°, IR + GPCP)                                NOAA/PERSIANN-CDR
Hasil dikirim ke web app J-FEW (action "rain_sat") → sheet JFEW_RAIN_SAT → "Dashboard Curah Hujan".
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT (opsional), JFEW_WEBAPP_URL, JFEW_BURN_TOKEN, RAIN_START (opsional, YYYY)
"""
import calendar
import datetime as dt
import os
import time

import ee

import jfew_burn_job as J

LON, LAT = 115.018279, -3.903238
START_YEAR = int(os.environ.get("RAIN_START") or 1998)


def months(y0):
    today = dt.date.today()
    out, y, m = [], y0, 1
    while (y, m) < (today.year, today.month):
        out.append((y, m)); m += 1
        if m == 13: y, m = y + 1, 1
    return out


def sample(img, scale):
    v = img.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([LON, LAT]), scale).values()
    return ee.Algorithms.If(v.size().gt(0), v.get(0), None)


def month_feature(y, m):
    s = ee.Date.fromYMD(y, m, 1); e = s.advance(1, "month"); hrs = calendar.monthrange(y, m)[1] * 24
    imerg_m = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").filterDate(s, e).select("precipitation")
    imerg_hh = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterDate(s, e).select("precipitation")
    gsm = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational").filterDate(s, e).select("hourlyPrecipRateGC")
    chp = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(s, e).select("precipitation")
    era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(s, e).select("total_precipitation_sum")
    per = ee.ImageCollection("NOAA/PERSIANN-CDR").filterDate(s, e).select("precipitation")
    d = {
        "ym": "%04d-%02d" % (y, m),
        # produk bulanan: laju rata-rata mm/jam × jam dalam bulan
        "imerg": ee.Algorithms.If(imerg_m.size().gt(0), ee.Number(sample(imerg_m.first(), 11132)).multiply(hrs),
                 # belum ada Final bulanan → jumlah half-hourly (mm/jam × 0,5 jam); harus lengkap ≥ 95 %
                 ee.Algorithms.If(imerg_hh.size().gte(calendar.monthrange(y, m)[1] * 48 * 0.95),
                                  ee.Number(sample(imerg_hh.sum(), 11132)).multiply(0.5), None)),
        # "final" = produk bulanan Final, atau half-hourly yang seluruhnya berstatus permanent (Final); selain itu "prov"
        "imerg_src": ee.Algorithms.If(imerg_m.size().gt(0), "final", ee.Algorithms.If(imerg_hh.size().gt(0),
                     ee.Algorithms.If(imerg_hh.filter(ee.Filter.neq("status", "permanent")).size().eq(0), "final", "prov"), "")),
        "gsmap": ee.Algorithms.If(gsm.size().gte(calendar.monthrange(y, m)[1] * 24 * 0.95), sample(gsm.sum(), 11132), None),
        "chirps": ee.Algorithms.If(chp.size().eq(6), sample(chp.sum(), 5566), None),
        "era5l": ee.Algorithms.If(era.size().gt(0), ee.Number(sample(era.first(), 11132)).multiply(1000), None),
        "persiann": ee.Algorithms.If(per.size().gte(calendar.monthrange(y, m)[1]), sample(per.sum(), 27830), None),
    }
    return ee.Feature(None, d)


def main():
    J.init_ee()
    rows, ms = [], months(START_YEAR)
    for i in range(0, len(ms), 12):                       # per tahun → aman dari batas waktu EE
        chunk = ms[i:i + 12]
        for att in range(3):
            try:
                fc = ee.FeatureCollection([month_feature(y, m) for y, m in chunk]).getInfo()
                break
            except Exception as ex:                            # noqa: BLE001
                print("ulang", chunk[0], ex); time.sleep(10 * (att + 1))
        else:
            raise SystemExit("gagal berulang di %s" % (chunk[0],))
        for f in fc["features"]:
            p = f["properties"]
            row = {k: (round(p[k], 1) if isinstance(p.get(k), (int, float)) else p.get(k)) for k in
                   ["ym", "imerg", "imerg_src", "gsmap", "chirps", "era5l", "persiann"]}
            rows.append(row)
        print(chunk[0][0], [r["imerg"] for r in rows[-len(chunk):]])
    print("bulan:", len(rows))
    for i in range(0, len(rows), 120):                     # kirim bertahap
        print(J.web({"action": "rain_sat", "rows": rows[i:i + 120], "final": i + 120 >= len(rows)}))


if __name__ == "__main__":
    main()
