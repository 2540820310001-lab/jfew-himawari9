"""
J-FEW · PESISIR & CUACA EKSTREM (jfew_coast.py · v1.1 · 13 Nov 2026 · + tiap sumber data dibungkus try, error ditulis di Summary)
Tiap 6 jam:
  1. Gelombang & pasang 7 hari di depan port (Open-Meteo Marine: wave_height, wave_period, wave_direction, sea_level_height_msl)
     kategori BMKG: tenang <0,5 · rendah 0,5–1,25 · sedang 1,25–2,5 · tinggi 2,5–4 · sangat tinggi 4–6 · ekstrem >6 m
  2. Angin kencang & potensi petir per jam 48 jam di dalam BBOX (Open-Meteo: wind_gusts_10m, weather_code, cape, precipitation)
  3. Gempa terkini & potensi tsunami BMKG (data.bmkg.go.id TEWS) + peringatan dini cuaca BMKG untuk Tanah Laut (bila tersedia)
Kirim ke peta: action "coast". Belum mengirim Telegram.
Ambang operasi (WAVE_LIMIT, GUST_WARN, GUST_STOP) adalah CONTOH — sesuaikan dengan SOP port & K3 PT JBG.
"""
import datetime as dt
import json
import math
import os
import re
import traceback
import urllib.request

import jfew_burn_job as J

WITA = dt.timezone(dt.timedelta(hours=8))
PORT = (114.970, -4.027)                                  # pusat poligon Office & Port (ZZZ_JFEW_BUFFER.gs)
SEA_PTS = [(114.985, -4.045), (114.99, -4.06), (115.0, -4.08), (114.97, -4.07)]   # titik laut di depan port (dipilih yang ada datanya)
BBOX_PTS = [(114.96, -3.88), (115.03, -3.90), (115.09, -3.89), (114.98, -4.00), (115.05, -3.98)]
WAVE_LIMIT = float(os.environ.get("WAVE_LIMIT") or 1.25)   # m — batas tongkang (contoh)
GUST_WARN = float(os.environ.get("GUST_WARN") or 40)       # km/jam waspada (contoh)
GUST_STOP = float(os.environ.get("GUST_STOP") or 60)       # km/jam hentikan crane/blasting/kerja di ketinggian (contoh)
WAVE_CAT = [(0.5, "tenang"), (1.25, "rendah"), (2.5, "sedang"), (4.0, "tinggi"), (6.0, "sangat tinggi"), (99, "ekstrem")]


def get(url, t=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "J-FEW coast"}), timeout=t).read().decode("utf-8", "replace")


def cat(h):
    return next(n for lim, n in WAVE_CAT if h < lim) if h is not None else None


def marine():
    for lon, lat in SEA_PTS:
        try:
            d = json.loads(get("https://marine-api.open-meteo.com/v1/marine?latitude=%s&longitude=%s&hourly=wave_height,wave_period,wave_direction,sea_level_height_msl"
                               "&timezone=Asia%%2FMakassar&forecast_days=7" % (lat, lon)))
            h = d["hourly"]
            if sum(1 for v in h.get("wave_height", []) if v is not None) > 24:
                return dict(lon=lon, lat=lat, time=h["time"], hs=h.get("wave_height"), tp=h.get("wave_period"), dir=h.get("wave_direction"), sl=h.get("sea_level_height_msl"))
        except Exception as ex:                                # noqa: BLE001
            print("marine", lon, lat, ex)
    return None


def weather48():
    lat = ",".join(str(p[1]) for p in BBOX_PTS); lon = ",".join(str(p[0]) for p in BBOX_PTS)
    d = json.loads(get("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&hourly=wind_gusts_10m,wind_speed_10m,weather_code,cape,precipitation"
                       "&timezone=Asia%%2FMakassar&forecast_days=3" % (lat, lon)))
    d = d if isinstance(d, list) else [d]
    t = d[0]["hourly"]["time"]; now = dt.datetime.now(WITA).strftime("%Y-%m-%dT%H:00")
    i0 = next((i for i, s in enumerate(t) if s >= now), 0); out = []
    for i in range(i0, min(i0 + 48, len(t))):
        g = max((x["hourly"]["wind_gusts_10m"][i] or 0) for x in d)
        wc = max((x["hourly"]["weather_code"][i] or 0) for x in d)
        cp = max((x["hourly"]["cape"][i] or 0) for x in d)
        pr = max((x["hourly"]["precipitation"][i] or 0) for x in d)
        petir = wc in (95, 96, 99) or (cp >= 1500 and pr >= 2)
        out.append(dict(t=t[i], gust=round(g), petir=bool(petir), hujan=round(pr, 1), cape=round(cp)))
    return out


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


def quake():
    out = {}
    try:
        g = json.loads(get("https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"))["Infogempa"]["gempa"]
        la, lo = [float(x) for x in g["Coordinates"].split(",")]
        pot = g.get("Potensi", "")
        out = dict(waktu="%s %s" % (g.get("Tanggal"), g.get("Jam")), mag=float(g.get("Magnitude")), kedalaman=g.get("Kedalaman"), wilayah=g.get("Wilayah"),
                   potensi=pot, jarak_km=round(km(la, lo, PORT[1], PORT[0])), lat=la, lon=lo,
                   tsunami=("tsunami" in pot.lower() and "tidak" not in pot.lower()))
    except Exception as ex:                                    # noqa: BLE001
        out = dict(error=str(ex)[:120])
    return out


def bmkg_warning():
    """peringatan dini cuaca BMKG (nowcast) yang menyebut Tanah Laut / Kalimantan Selatan — opsional"""
    try:
        x = get("https://www.bmkg.go.id/alerts/nowcast/id/rss.xml", 30)
        items = re.findall(r"<item>(.*?)</item>", x, re.S); out = []
        for it in items:
            t = re.sub(r"<[^>]+>|<!\[CDATA\[|\]\]>", " ", it)
            if "Kalimantan Selatan" in t or "Tanah Laut" in t:
                title = re.search(r"<title>(.*?)</title>", it, re.S)
                out.append(re.sub(r"<!\[CDATA\[|\]\]>", "", title.group(1)).strip() if title else t.strip()[:160])
        return out[:3]
    except Exception as ex:                                    # noqa: BLE001
        print("peringatan BMKG tidak tersedia:", str(ex)[:100]); return None


def main():
    now = dt.datetime.now(WITA)
    errs = []

    def safe(name, fn, default):
        try:
            return fn()
        except Exception as ex:                                # noqa: BLE001
            errs.append("%s: %s" % (name, str(ex)[:200])); print("GAGAL", name); traceback.print_exc()
            return default
    M = safe("gelombang (Open-Meteo Marine)", marine, None)
    W = safe("angin & petir (Open-Meteo)", weather48, [])
    Q = safe("gempa BMKG", quake, {})
    B = safe("peringatan BMKG", bmkg_warning, None)
    res = dict(issued=now.strftime("%Y-%m-%d %H:%M"), port=PORT, limits=dict(wave=WAVE_LIMIT, gust_warn=GUST_WARN, gust_stop=GUST_STOP), wind=W, quake=Q, bmkg=B)
    if M:
        hs = [v for v in M["hs"] if v is not None]
        nowi = next((i for i, s in enumerate(M["time"]) if s >= now.strftime("%Y-%m-%dT%H:00")), 0)
        sl = M["sl"] or []
        imax = max((i for i in range(len(sl)) if sl[i] is not None), key=lambda i: sl[i], default=None)
        ihs = max(range(nowi, len(M["hs"])), key=lambda i: M["hs"][i] or 0)
        res["marine"] = dict(lon=M["lon"], lat=M["lat"], time=M["time"], hs=M["hs"], sl=M["sl"], tp=M["tp"], dir=M["dir"],
                             now=M["hs"][nowi], now_cat=cat(M["hs"][nowi]), max=M["hs"][ihs], max_t=M["time"][ihs], max_cat=cat(M["hs"][ihs]),
                             tide_max=None if imax is None else round(sl[imax], 2), tide_max_t=None if imax is None else M["time"][imax],
                             jam_aman=sum(1 for v in M["hs"][nowi:] if v is not None and v < WAVE_LIMIT), jam_total=len(M["hs"][nowi:]))
    g = max(W, key=lambda r: r["gust"]) if W else None
    res["gust_max"] = g
    warn = []
    if B: warn += [dict(level="merah", teks="BMKG: " + b) for b in B]
    if g and g["gust"] >= GUST_STOP: warn.append(dict(level="merah", teks="Hembusan angin %d km/jam %s — hentikan crane, blasting & kerja di ketinggian" % (g["gust"], g["t"].replace("T", " "))))
    elif g and g["gust"] >= GUST_WARN: warn.append(dict(level="oranye", teks="Hembusan angin %d km/jam %s — waspada" % (g["gust"], g["t"].replace("T", " "))))
    pt = [r for r in W if r["petir"]]
    if pt: warn.append(dict(level="oranye", teks="Potensi petir %d jam dalam 48 jam (mulai %s)" % (len(pt), pt[0]["t"].replace("T", " "))))
    mm = res.get("marine")
    if mm and mm["max"] is not None and mm["max"] >= WAVE_LIMIT:
        warn.append(dict(level="oranye", teks="Gelombang %s m (%s) %s — batasi/tunda tongkang" % (round(mm["max"], 1), mm["max_cat"], mm["max_t"].replace("T", " "))))
    if Q.get("mag") is not None:
        warn.append(dict(level="merah" if Q["tsunami"] else "abu", teks="Gempa terakhir M%.1f · %s km dari port · %s" % (Q["mag"], Q["jarak_km"], Q["potensi"] or "")))
    res["warn"] = warn
    res["errors"] = errs
    try:
        print(J.web({"action": "coast", "result": res}))
    except Exception as ex:                                    # noqa: BLE001
        msg = str(ex)
        hint = ""
        if "action tidak dikenal" in msg:
            hint = "Apps Script belum versi baru: ganti ZZZ_JFEW_BURN_AUTO.gs (v1.16) lalu Kelola deployment → ✏️ → Versi baru."
        elif "JFEW_COAST_SAVE_" in msg or "is not defined" in msg:
            hint = "File ZZZ_JFEW_COAST.gs belum ditambahkan di Apps Script, atau deploy belum versi baru."
        elif "token" in msg:
            hint = "Secret JFEW_BURN_TOKEN tidak cocok dengan Script Property JFEW_BURN_TOKEN."
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("# J-FEW pesisir — GAGAL kirim ke peta\n\n`%s`\n\n%s\n\nSumber data bermasalah: %s\n" % (msg[:400], hint, "; ".join(errs) or "tidak ada"))
        raise
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        mm = res.get("marine") or {}
        md = ["# J-FEW · pesisir & cuaca ekstrem", "", "Gelombang sekarang %s m (%s) · maks 7 hari %s m (%s, %s) · pasang tertinggi %s m (%s) · jam aman tongkang %s/%s" %
              (mm.get("now"), mm.get("now_cat"), mm.get("max"), mm.get("max_cat"), mm.get("max_t"), mm.get("tide_max"), mm.get("tide_max_t"), mm.get("jam_aman"), mm.get("jam_total")),
              "", "Angin maks 48 jam: %s km/jam (%s)" % (g and g["gust"], g and g["t"]), "", "Peringatan:"] + ["- " + w["teks"] for w in warn] + \
             (["", "Sumber data bermasalah (panel tetap jalan):"] + ["- " + e for e in errs] if errs else [])
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))


if __name__ == "__main__":
    main()
