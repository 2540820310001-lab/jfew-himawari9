"""
J-FEW · HUJAN EKSTREM & SKENARIO GENANGAN (jfew_rain_extreme.py · v1.2 · hasil hujan tetap tersimpan walau simulasi gagal · v1.1 · 12 Nov 2026 · potongan data per bulan, pesan galat lebih jelas)
1. Hujan harian 2001–sekarang dari satelit (Earth Engine):
     IMERG V07 di titik ombrometer & rata-rata area tambang, CHIRPS v2 rata-rata area
   dikoreksi ke ombrometer PT JBG (rasio kuantil ekor atas pada hari yang sama).
2. Maksimum tahunan hujan 1, 3, 5 hari → analisis frekuensi (Normal, Log Normal, Gumbel, Log-Pearson III; uji KS & Chi²)
   → hujan rencana R2–R100 dan periode ulang hujan 250 mm/hari.
3. Daftar 20 kejadian hujan terbesar (bahan kalibrasi dengan genangan Sentinel-1 / luas air void Landsat-Sentinel-2).
4. Simulasi genangan rain-on-grid (jfew_flood_sim.py) untuk skenario hujan 3 hari R5, R25, R100 → status void & genangan.
Keluaran: ringkasan di halaman Summary GitHub, artefak extreme_out/ (JSON, CSV, PNG).
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT, JFEW_WEBAPP_URL, JFEW_BURN_TOKEN · opsional SCENARIOS="5,25,100", CN=85, START=2001
"""
import datetime as dt
import json
import math
import os

import numpy as np
import pandas as pd

import jfew_rain_forecast as F          # freq(), get_input(), LAT/LON titik ombrometer

OUT = "extreme_out"
START = int(os.environ.get("START") or 2001)
AOI = [114.96, -3.915, 115.092, -3.862]          # area tambang (cakupan topo EOM)


# ------------------------------------------------------------------ data hujan harian
def satellite_daily():
    import ee
    import jfew_burn_job as J
    J.init_ee()
    pt = ee.Geometry.Point([F.LON, F.LAT]); area = ee.Geometry.Rectangle(AOI)
    im = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
    ch = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    rows = []
    last = dt.date.today() - dt.timedelta(days=5)
    for y in range(START, last.year + 1):
        n = ((dt.date(y + 1, 1, 1) if y < last.year else last) - dt.date(y, 1, 1)).days
        def feat(k):
            d = ee.Date.fromYMD(y, 1, 1).advance(k, "day")
            s = d.advance(-8, "hour")                                          # hari WITA
            i = im.filterDate(s, s.advance(1, "day")).sum().multiply(0.5)
            c = ch.filterDate(d, d.advance(1, "day")).first()
            c = ee.Image(ee.Algorithms.If(c, c, ee.Image.constant(-1).rename("precipitation")))
            return ee.Feature(None, {"d": d.format("YYYY-MM-dd"),
                                     "ip": i.reduceRegion(ee.Reducer.first(), pt, 11132).values().get(0),
                                     "ia": i.reduceRegion(ee.Reducer.mean(), area, 11132).values().get(0),
                                     "ca": c.reduceRegion(ee.Reducer.mean(), area, 5566).values().get(0)})
        for a in range(0, n, 31):                                             # potongan ±1 bulan (batas memori/waktu Earth Engine)
            fc = ee.FeatureCollection(ee.List.sequence(a, min(n, a + 31) - 1).map(feat))
            res = None
            for k in range(5):
                try:
                    res = fc.getInfo(); break
                except Exception as ex:                                       # noqa: BLE001
                    print("  ulang %d (%s hari %d): %s" % (k + 1, y, a, str(ex)[:200]), flush=True); import time; time.sleep(15 * (k + 1))
            if res is None:
                raise RuntimeError("Earth Engine gagal untuk %d hari ke-%d" % (y, a))
            for f in res["features"]:
                p = f["properties"]
                rows.append([p["d"], p.get("ip"), p.get("ia"), None if (p.get("ca") is None or p.get("ca") < 0) else p.get("ca")])
        print("satelit", y, "hari", sum(1 for r in rows if r[0].startswith(str(y))), flush=True)
    return pd.DataFrame(rows, columns=["d", "imerg_titik", "imerg_area", "chirps_area"]).set_index("d").astype(float)


def correction(S, G):
    """rasio kuantil P95–P99,5 ombrometer / satelit pada hari basah yang sama (dibatasi 0,7–3)"""
    out = {}
    j = S.join(G, how="inner")
    for c in ["imerg_titik", "imerg_area", "chirps_area"]:
        w = j[(j["gauge"] > 1) | (j[c] > 1)].dropna(subset=[c, "gauge"])
        if len(w) < 100:
            out[c] = 1.0; continue
        qs = np.arange(0.95, 0.996, 0.005)
        out[c] = float(np.clip(np.median([np.quantile(w["gauge"], q) / max(np.quantile(w[c], q), 0.1) for q in qs]), 0.7, 3.0))
    return out, int(len(j))


def annual_max(s, k):
    r = s.rolling(k, min_periods=k).sum()
    yr = r.index.str[:4].astype(int)
    full = pd.Series(s.notna().values, index=s.index).groupby(s.index.str[:4].astype(int)).sum()
    am = r.groupby(yr).max()
    return am[[y for y in am.index if full.get(y, 0) >= 330]]               # hanya tahun yang datanya hampir lengkap


def return_period(fr, v):
    """periode ulang (tahun) nilai v dari distribusi terpilih (interpolasi log)"""
    b = [d for d in fr["dist"] if d["name"] == fr["best"]][0]; R = b["R"]; T = fr["T"]
    if v <= R[0]: return "< 2"
    if v >= R[-1]: return "> 100"
    for i in range(len(R) - 1):
        if R[i] <= v <= R[i + 1]:
            lt = math.log(T[i]) + (math.log(T[i + 1]) - math.log(T[i])) * (v - R[i]) / (R[i + 1] - R[i])
            return "%.0f" % math.exp(lt)


def top_events(s, k, n=20, sep=5):
    r = s.rolling(k, min_periods=k).sum().dropna().sort_values(ascending=False)
    picked = []
    for d, v in r.items():
        dd = dt.date.fromisoformat(d)
        if all(abs((dd - p[0]).days) > sep for p in picked):
            picked.append((dd, v))
        if len(picked) >= n: break
    return picked


def design_hyeto(R1, R3):
    """hujan 3 hari: hari ke-2 = R1 (pola Mononobe, puncak jam ke-4); sisa (R3−R1) merata di hari 1 & 3"""
    t = np.arange(1, 25); depth = R1 / 24 * (24 / t) ** (2 / 3); inc = np.diff(np.r_[0, depth])
    order = [3]; l, r = 2, 4
    while len(order) < 24:
        if r < 24: order.append(r); r += 1
        if l >= 0 and len(order) < 24: order.append(l); l -= 1
    d2 = np.zeros(24); d2[order] = inc
    rest = max(R3 - R1, 0) / 48
    return np.r_[np.full(24, rest), d2, np.full(24, rest)]


def main():
    os.makedirs(OUT, exist_ok=True)
    I = F.get_input()
    G = pd.DataFrame(I.get("daily") or [], columns=["d", "gauge", "du", "sl"]).set_index("d")[["gauge"]].astype(float)
    S = satellite_daily()
    fac, npair = correction(S, G)
    C = S.copy()
    for c in fac: C[c] = S[c] * fac[c]
    C.to_csv(OUT + "/hujan_harian_terkoreksi.csv")
    md = ["# J-FEW · hujan ekstrem & skenario genangan", "",
          "Hujan harian satelit %s–%s dikoreksi ke ombrometer PT JBG (%d hari berpasangan). Faktor koreksi: %s." %
          (S.index.min(), S.index.max(), npair, ", ".join("%s ×%.2f" % (k, v) for k, v in fac.items())), ""]
    # ---- frekuensi
    FR = {}
    for c, lab in [("imerg_titik", "IMERG titik ombrometer"), ("imerg_area", "IMERG rata-rata area tambang"), ("chirps_area", "CHIRPS rata-rata area")]:
        for k in (1, 3, 5):
            am = annual_max(C[c].dropna(), k)
            fr = F.freq(list(am.values))
            if fr:
                fr["years"] = "%d–%d" % (am.index.min(), am.index.max()); fr["amax"] = {int(y): round(float(v), 1) for y, v in am.items()}
                FR["%s_%dd" % (c, k)] = fr
    ga = annual_max(G["gauge"].dropna(), 1)
    md += ["## Hujan rencana (distribusi terpilih per seri)", "", "| Seri | Durasi | Tahun | Distribusi | R2 | R5 | R10 | R25 | R50 | R100 | Maks tercatat |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, fr in FR.items():
        b = [d for d in fr["dist"] if d["name"] == fr["best"]][0]
        c, k = key.rsplit("_", 1)
        md.append("| %s | %s | %s | %s | %s | %.0f |" % (c.replace("_", " "), k.replace("d", " hari"), fr["years"], fr["best"], " | ".join("%.0f" % v for v in b["R"]), max(fr["amax"].values())))
    main_key = "imerg_titik_1d"
    if main_key in FR:
        md += ["", "**Hujan 250 mm/hari** setara periode ulang **±%s tahun** (IMERG titik ombrometer terkoreksi, %s)." % (return_period(FR[main_key], 250), FR[main_key]["best"])]
    if len(ga):
        md += ["Maksimum harian ombrometer PT JBG yang tercatat: %s." % ", ".join("%d: %.0f mm" % (y, v) for y, v in ga.sort_values(ascending=False).head(5).items())]
    # ---- kejadian terbesar
    ev = []
    for k in (1, 3):
        for d, v in top_events(C["imerg_titik"].dropna(), k):
            ds = d.isoformat(); g = G["gauge"].get(ds)
            ev.append(dict(durasi_hari=k, tanggal_akhir=ds, imerg_mm=round(float(v), 0), ombrometer_mm=None if g is None or k > 1 else float(g),
                           chirps_mm=round(float(C["chirps_area"].rolling(k).sum().get(ds, float("nan"))), 0)))
    pd.DataFrame(ev).to_csv(OUT + "/kejadian_hujan_terbesar.csv", index=False)
    md += ["", "## 10 kejadian hujan 1 hari terbesar (bahan kalibrasi Sentinel-1)", "", "| Tanggal | IMERG terkoreksi | Ombrometer | CHIRPS |", "|---|---|---|---|"]
    md += ["| %s | %.0f | %s | %s |" % (e["tanggal_akhir"], e["imerg_mm"], "–" if e["ombrometer_mm"] is None else "%.0f" % e["ombrometer_mm"], e["chirps_mm"]) for e in ev if e["durasi_hari"] == 1][:10]
    # ---- skenario genangan
    scn = [int(v) for v in (os.environ.get("SCENARIOS") or "5,25,100").split(",")]
    CN = float(os.environ.get("CN") or 85)
    res = {"faktor": fac, "frekuensi": FR, "kejadian": ev, "skenario": {}}
    json.dump(res, open(OUT + "/hasil_hujan_ekstrem.json", "w"), default=float, indent=1)        # simpan dulu hasil hujan
    open(OUT + "/ringkasan.md", "w").write("\n".join(md))
    Gm = None
    try:
        import jfew_flood_sim as M
        Gm = M.load()
    except Exception as ex:                                                    # noqa: BLE001
        md += ["", "## ⚠ Simulasi genangan dilewati", "", "```", str(ex)[:1500], "```"]
    if Gm is not None and "imerg_titik_1d" in FR and "imerg_titik_3d" in FR:
        b1 = [d for d in FR["imerg_titik_1d"]["dist"] if d["name"] == FR["imerg_titik_1d"]["best"]][0]["R"]
        b3 = [d for d in FR["imerg_titik_3d"]["dist"] if d["name"] == FR["imerg_titik_3d"]["best"]][0]["R"]
        md += ["", "## Skenario genangan (hujan 3 hari, topo EOM Agu 2026, CN %.0f)" % CN, ""]
        for T in scn:
            i = F.T_RET.index(T); R1, R3 = b1[i], max(b3[i], b1[i])
            print("skenario R%d: 1 hari %.0f mm, 3 hari %.0f mm" % (T, R1, R3), flush=True)
            R = M.run(Gm, design_hyeto(R1, R3), CN=CN)
            Sm = M.summarize(Gm, R)
            M.plot(Gm, R, Sm, "R%d · hujan 3 hari %.0f mm (puncak harian %.0f mm) · genangan maksimum & status void" % (T, R3, R1), OUT + "/genangan_R%d.png" % T)
            res["skenario"]["R%d" % T] = dict(R1=R1, R3=R3, **Sm)
            md += ["### R%d — hujan 3 hari %.0f mm (hari terbesar %.0f mm)" % (T, R3, R1), "",
                   "Luas tergenang (di luar void & sungai): " + " · ".join("%s %.1f ha" % kv for kv in Sm["area_ha"].items()), "",
                   "| Void | MA awal | Naik | MA akhir | Spillway | Sisa | Status |", "|---|---|---|---|---|---|---|"]
            md += ["| %s | %s | +%.2f | %.2f | %s | %s | %s |" % (v["void"], v["ma_awal"], v["naik_m"], v["ma_akhir"], v["spill"], v["sisa_m"], v["status"]) for v in Sm["voids"]]
            md += ["", "Genangan terbesar: " + "; ".join("%s %.1f ha (maks %.1f m)" % (s["lokasi"], s["luas_ha"], s["dalam_maks_m"]) for s in Sm["spots"][:5]), ""]
    json.dump(res, open(OUT + "/hasil_hujan_ekstrem.json", "w"), default=float, indent=1)
    md += ["", "_Konsep: muka air void awal dari data void J-FEW (31/10/2026); kenaikan dihitung dengan dinding void tegak (sedikit berlebih). "
           "Admin, port & Z04 di luar cakupan topo. Pompa, gorong-gorong dan banjir kiriman hulu belum dimodelkan._"]
    open(OUT + "/ringkasan.md", "w").write("\n".join(md))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:                                                    # noqa: BLE001
        import traceback; tb = traceback.format_exc(); print(tb)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("## ❌ Gagal\n\n```\n%s\n```\n" % tb[-3000:])
        raise
