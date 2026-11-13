"""
J-FEW · PRAKIRAAN MUSIM (jfew_season.py · v1.1 · 13 Nov 2026 · + normal bulanan & hujan rencana 3 hari untuk panel)
Sebulan sekali (tgl 12, setelah NOAA CPC terbit Kamis ke-2) atau manual:
  1. ENSO & IOD teramati: Niño3.4 & DMI bulanan (NOAA PSL)
  2. Prakiraan ENSO resmi: tabel peluang kekuatan Niño-3.4 NOAA CPC (ENSO strengths, per musim 3-bulanan)
  3. Prakiraan hujan 6 bulan di titik ombrometer JBG: ECMWF SEAS5 (Open-Meteo Seasonal, semua anggota ensemble)
     dibandingkan dengan normal bulanan 2001–2025 (ombrometer bila ada, selain itu IMERG) dari modul curah hujan J-FEW
  4. Awal musim per dasarian (kriteria BMKG: hujan dasarian ≥ 50 mm = musim hujan, < 50 mm = kemarau, berturut 3 dasarian)
  5. Pandangan musim kebakaran: tahun analog ENSO/IOD 2001–2025 + jumlah hotspot MODIS per tahun (ML tahap 1)
Kirim ke peta: action "season". Belum mengirim Telegram.
Env: JFEW_WEBAPP_URL, JFEW_BURN_TOKEN (EE tidak diperlukan)
"""
import datetime as dt
import json
import os
import re
import urllib.request

import numpy as np

import jfew_burn_job as J

LAT, LON = -3.903238, 115.018279
WITA = dt.timezone(dt.timedelta(hours=8))
SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]   # musim berpusat bulan 1..12
# piksel-hari hotspot MODIS di kotak J-FEW per tahun (jfew_ml_dataset.py, run 11 Nov 2026)
FIRE = {2001: 12, 2002: 394, 2003: 454, 2004: 490, 2005: 167, 2006: 374, 2007: 19, 2008: 9, 2009: 100, 2010: 13, 2011: 86, 2012: 99,
        2013: 99, 2014: 210, 2015: 646, 2016: 21, 2017: 0, 2018: 69, 2019: 184, 2020: 4, 2021: 0, 2022: 9, 2023: 124, 2024: 3, 2025: 0}
HEAVY = 150          # musim api "berat" bila ≥ 150 piksel-hari


def fetch(url, timeout=90):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "J-FEW season"}), timeout=timeout).read().decode("utf-8", "replace")


def psl(url):
    s = {}
    for ln in fetch(url).splitlines():
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


def cpc_strengths():
    """tabel peluang (%) Niño-3.4 per musim: kolom ≤-2,≤-1.5,≤-1,≤-0.5,≥0.5,≥1,≥1.5,≥2"""
    html = fetch("https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/strengths/")
    txt = re.sub(r"<[^>]+>", " ", html); txt = re.sub(r"&[a-z]+;", " ", txt)
    rows = []
    for m in re.finditer(r"\b(%s)\b((?:\s+~?\d+){8})" % "|".join(SEAS), txt):
        v = [0 if t.startswith("~") else int(t) for t in m.group(2).split()]
        rows.append({"musim": m.group(1), "p_lanina": v[3], "p_elnino": v[4], "p_netral": max(0, 100 - v[3] - v[4]),
                     "p_kuat_elnino": v[6], "p_kuat_lanina": v[1], "raw": v})
    issued = re.search(r"(?:updated|issued)[^.]{0,80}?(\d{1,2}\s+\w+\s+20\d\d|\w+\s+\d{1,2},\s+20\d\d)", txt, re.I)
    return rows, (issued.group(1) if issued else None)


def seas5_daily():
    url = ("https://seasonal-api.open-meteo.com/v1/seasonal?latitude=%s&longitude=%s&daily=precipitation_sum&models=ecmwf_seas5"
           "&timezone=Asia%%2FMakassar&forecast_days=183" % (LAT, LON))
    d = json.loads(fetch(url, 120))["daily"]
    t = d["time"]; mem = [np.array(v, float) for k, v in d.items() if k.startswith("precipitation_sum")]
    return t, np.array(mem)                                            # [anggota, hari]


def normals(I):
    """normal bulanan 2001–2025 (mm) dari seri bulanan J-FEW: ombrometer bila valid, selain itu satelit"""
    by = {m: [] for m in range(1, 13)}
    for r in I.get("months") or []:
        ym, g, s = r[0], r[1], r[2]
        y, m = int(str(ym)[:4]), int(str(ym)[5:7])
        if 2001 <= y <= 2025:
            v = g if g is not None else s
            if v is not None: by[m].append(float(v))
    return {m: dict(mean=round(float(np.mean(v)), 0), p25=round(float(np.percentile(v, 25)), 0), p75=round(float(np.percentile(v, 75)), 0), n=len(v))
            for m, v in by.items() if len(v) >= 5}


def seasonal_mean(s, y, m):
    v = [s.get((y + (mm - 1) // 12, (mm - 1) % 12 + 1)) for mm in (m - 1, m, m + 1)]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if len(v) == 3 else None


def main():
    now = dt.datetime.now(WITA)
    nino = psl("https://psl.noaa.gov/data/correlation/nina34.anom.data")
    dmi = psl("https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data")
    last_n = max(nino); last_d = max(dmi)
    obs = {"nino34": round(nino[last_n], 2), "nino34_bulan": "%d-%02d" % last_n,
           "nino34_3bln": [round(nino.get(k, float("nan")), 2) for k in sorted(nino)[-6:]], "nino34_bln_list": ["%d-%02d" % k for k in sorted(nino)[-6:]],
           "dmi": round(dmi[last_d], 2), "dmi_bulan": "%d-%02d" % last_d,
           "dmi_3bln": [round(dmi.get(k, float("nan")), 2) for k in sorted(dmi)[-6:]], "dmi_bln_list": ["%d-%02d" % k for k in sorted(dmi)[-6:]]}
    n = obs["nino34"]
    obs["enso_status"] = ("El Niño sangat kuat" if n >= 2 else "El Niño kuat" if n >= 1.5 else "El Niño sedang" if n >= 1 else "El Niño lemah" if n >= 0.5 else
                          "La Niña sangat kuat" if n <= -2 else "La Niña kuat" if n <= -1.5 else "La Niña sedang" if n <= -1 else "La Niña lemah" if n <= -0.5 else "Netral")
    d = obs["dmi"]
    obs["iod_status"] = "IOD positif" if d >= 0.4 else "IOD negatif" if d <= -0.4 else "IOD netral"
    try:
        cpc, cpc_issued = cpc_strengths()
    except Exception as ex:                                            # noqa: BLE001
        print("CPC gagal:", ex); cpc, cpc_issued = [], None
    # ---- hujan 6 bulan
    I = J.web({"action": "rain_input"}) or {}
    NOR = normals(I)
    rain = []; onset = []
    try:
        t, mem = seas5_daily()
        months = sorted({s[:7] for s in t})
        for ym in months:
            idx = [i for i, s in enumerate(t) if s.startswith(ym)]
            if len(idx) < 25: continue
            tot = mem[:, idx].sum(axis=1); m = int(ym[5:7]); nr = NOR.get(m, {})
            med = float(np.median(tot))
            rain.append(dict(bulan=ym, median=round(med), p25=round(float(np.percentile(tot, 25))), p75=round(float(np.percentile(tot, 75))),
                             normal=nr.get("mean"), normal_p25=nr.get("p25"), normal_p75=nr.get("p75"),
                             pct=None if not nr.get("mean") else round((med / nr["mean"] - 1) * 100),
                             p_bawah_normal=None if not nr.get("p25") else round(float((tot < nr["p25"]).mean() * 100)),
                             p_atas_normal=None if not nr.get("p75") else round(float((tot > nr["p75"]).mean() * 100))))
        # dasarian (1–10, 11–20, 21–akhir) — median ensemble
        das = {}
        for i, s in enumerate(t):
            k = "%s-D%d" % (s[:7], 1 if int(s[8:10]) <= 10 else 2 if int(s[8:10]) <= 20 else 3)
            das.setdefault(k, []).append(i)
        keys = sorted(das)
        tot = {k: float(np.median(mem[:, das[k]].sum(axis=1))) for k in keys}
        state = None
        for i, k in enumerate(keys[:-2]):
            trio = [tot[keys[i + j]] for j in range(3)]
            s_new = "hujan" if all(v >= 50 for v in trio) else "kemarau" if all(v < 50 for v in trio) else None
            if s_new and s_new != state:
                onset.append(dict(dasarian=k, musim=s_new)); state = s_new
        das_list = [dict(dasarian=k, mm=round(tot[k])) for k in keys]
        seas_src = "ECMWF SEAS5 via Open-Meteo · %d anggota · %s s.d. %s" % (mem.shape[0], t[0], t[-1])
    except Exception as ex:                                            # noqa: BLE001
        print("SEAS5 gagal:", ex); das_list = []; seas_src = "gagal: %s" % str(ex)[:120]
    # ---- pandangan musim kebakaran (Agu–Okt): analog ENSO (JAS) & IOD (SON)
    hist = []
    for y, f in FIRE.items():
        e = seasonal_mean(nino, y, 8); i_ = seasonal_mean(dmi, y, 10)
        hist.append(dict(tahun=y, api=f, nino_jas=None if e is None else round(e, 2), dmi_son=None if i_ is None else round(i_, 2),
                         kategori="El Niño" if (e or 0) >= 0.5 else "La Niña" if (e or 0) <= -0.5 else "Netral", iod_pos=(i_ or 0) >= 0.4))
    def p_heavy(rows):
        return None if not rows else round(100 * sum(1 for r in rows if r["api"] >= HEAVY) / len(rows))
    cond = {c: dict(n=len([r for r in hist if r["kategori"] == c]), p_berat=p_heavy([r for r in hist if r["kategori"] == c]),
                    rata_api=round(float(np.mean([r["api"] for r in hist if r["kategori"] == c] or [0]))),
                    tahun=[r["tahun"] for r in hist if r["kategori"] == c]) for c in ("El Niño", "Netral", "La Niña")}
    cond["IOD positif"] = dict(n=len([r for r in hist if r["iod_pos"]]), p_berat=p_heavy([r for r in hist if r["iod_pos"]]),
                               tahun=[r["tahun"] for r in hist if r["iod_pos"]])
    tgt = next((c for c in cpc if c["musim"] == "JAS"), None) or (cpc[-1] if cpc else None)
    fire = None
    if tgt:
        p = (tgt["p_elnino"] * (cond["El Niño"]["p_berat"] or 0) + tgt["p_netral"] * (cond["Netral"]["p_berat"] or 0) + tgt["p_lanina"] * (cond["La Niña"]["p_berat"] or 0)) / 100
        fire = dict(musim_acuan=tgt["musim"], p_elnino=tgt["p_elnino"], p_netral=tgt["p_netral"], p_lanina=tgt["p_lanina"], p_berat=round(p),
                    tingkat="tinggi" if p >= 50 else "sedang" if p >= 25 else "rendah",
                    analog=sorted([r for r in hist if r["kategori"] == ("El Niño" if tgt["p_elnino"] >= max(tgt["p_netral"], tgt["p_lanina"]) else "La Niña" if tgt["p_lanina"] >= tgt["p_netral"] else "Netral")],
                                  key=lambda r: -r["api"])[:4])
    res = dict(issued=now.strftime("%Y-%m-%d %H:%M"), normal={str(k): v for k, v in NOR.items()},
               r3_rencana={"T": [2, 5, 10, 25, 50, 100], "R3": [214, 264, 290, 318, 336, 352]},   # hujan 3 hari IMERG terkoreksi (jfew_rain_extreme.py)
               obs=obs, cpc=cpc, cpc_issued=cpc_issued, rain=rain, seas5=seas_src,
               dasarian=das_list, onset=onset, fire=fire, cond=cond, hist=hist, heavy=HEAVY)
    print(J.web({"action": "season", "result": res}))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        md = ["# J-FEW · prakiraan musim", "",
              "**ENSO**: Niño3.4 %s = %+.2f °C → %s · **IOD**: DMI %s = %+.2f °C → %s" % (obs["nino34_bulan"], obs["nino34"], obs["enso_status"], obs["dmi_bulan"], obs["dmi"], obs["iod_status"]), "",
              "## Peluang ENSO (NOAA CPC, %s)" % (cpc_issued or "terbaru"), "", "| Musim | El Niño | Netral | La Niña | El Niño kuat (≥1,5) |", "|---|---|---|---|---|"]
        md += ["| %s | %d%% | %d%% | %d%% | %d%% |" % (c["musim"], c["p_elnino"], c["p_netral"], c["p_lanina"], c["p_kuat_elnino"]) for c in cpc]
        md += ["", "## Hujan bulanan (%s)" % seas_src, "", "| Bulan | Prakiraan (median, P25–P75) | Normal | Selisih | P bawah normal | P atas normal |", "|---|---|---|---|---|---|"]
        md += ["| %s | %s (%s–%s) | %s | %s%% | %s%% | %s%% |" % (r["bulan"], r["median"], r["p25"], r["p75"], r["normal"], r["pct"], r["p_bawah_normal"], r["p_atas_normal"]) for r in rain]
        md += ["", "Awal musim (dasarian, kriteria 50 mm): " + (" → ".join("%s %s" % (o["musim"], o["dasarian"]) for o in onset) or "–")]
        if fire:
            md += ["", "## Musim kebakaran (acuan %s): peluang musim berat **%s%% (%s)**" % (fire["musim_acuan"], fire["p_berat"], fire["tingkat"]),
                   "Peluang musim api berat per kondisi 2001–2025: " + " · ".join("%s %s%% (n=%s)" % (k, v["p_berat"], v["n"]) for k, v in cond.items())]
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))


if __name__ == "__main__":
    main()
