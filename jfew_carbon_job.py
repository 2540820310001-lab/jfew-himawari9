
#!/usr/bin/env python3
"""
J-FEW · Estimasi EMISI KARBON kebakaran per boundary lapangan (drone) — jfew_carbon_job.py (v1 · 8 Nov 2026)
-------------------------------------------------------------------------------------------------------------
Emisi = Σ_kelas  A_kelas × B × CC_kelas × EF          (IPCC 2006 Vol.4 Pers. 2.27 / Seiler & Crutzen 1980)
  A   = luas per kelas keparahan dNBR di DALAM boundary drone (ha)  → drone_reference.geojson
  B   = biomassa di atas tanah SEBELUM kebakaran (t bahan kering/ha), dari satelit, urutan prioritas:
          1) GEDI L4A (lidar, footprint 25 m, kualitas baik, ≥ 10 tembakan, s.d. sebelum tanggal api)
          2) ESA CCI Biomass (100 m, awesome-gee-community-catalog)
          3) GEDI L4B (grid 1 km)
          4) nilai cadangan (FALLBACK_AGB) — ditandai jelas
  CC  = fraksi terbakar per kelas keparahan dNBR (ASUMSI, lihat CC_BY_CLASS)
  EF  = faktor emisi (g/kg bahan kering) & GWP — ASUMSI, lihat EF_G_PER_KG & GWP (VERIFIKASI ke sumber primer)
Hasil dikirim ke web app J-FEW (action "carbon_result") → sheet JFEW_CARBON → Dashboard Lokasi kolom Emisi.
Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT (opsional), JFEW_WEBAPP_URL, JFEW_BURN_TOKEN, JFEW_GWP (AR5 | AR6)
"""
import datetime as dt
import json
import os

import ee

import jfew_burn_job as J

# ---------- ASUMSI (semua perlu diverifikasi ke sumber primer sebelum dipakai untuk pelaporan resmi) ----------
# Faktor emisi "tropical forest", IPCC 2006 Vol.4 Bab 2 Tabel 2.5 (g gas per kg bahan kering terbakar)
EF_G_PER_KG = {"CO2": 1580.0, "CH4": 6.8, "N2O": 0.20}
GWP = {"AR5": {"CO2": 1, "CH4": 28, "N2O": 265}, "AR6": {"CO2": 1, "CH4": 27.0, "N2O": 273}}
# Fraksi biomassa di atas tanah yang terbakar per kelas dNBR. Api permukaan di bawah tajuk reklamasi umumnya
# hanya menghabiskan serasah & tumbuhan bawah → CC kecil pada kelas rendah.
CC_BY_CLASS = [(-9, 0.10, 0.10, "sangat rendah (< 0,10)"), (0.10, 0.27, 0.20, "rendah"), (0.27, 0.44, 0.35, "sedang-rendah"),
               (0.44, 0.66, 0.50, "sedang-tinggi"), (0.66, 9, 0.65, "tinggi")]
FALLBACK_AGB = 60.0          # t/ha — cadangan bila semua data satelit gagal (hutan sekunder muda; ASUMSI)
UNC_LOW, UNC_HIGH = 0.6, 1.5  # rentang ketidakpastian gabungan (×)
ESA_CCI_ASSETS = ["projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"]


def agb_sources(geom, before):
    out = {}
    try:
        ic = (ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY").filterBounds(geom.buffer(300))
              .filterDate("2019-04-01", before))
        def q(i):
            return i.updateMask(i.select("l4_quality_flag").eq(1).And(i.select("degrade_flag").eq(0))).select("agbd")
        m = ic.map(q).mean()
        r = m.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), "", True), geom.buffer(300), 25, maxPixels=1e9).getInfo()
        if r.get("agbd_count", 0) >= 10 and r.get("agbd_mean") is not None:
            out["GEDI L4A"] = (float(r["agbd_mean"]), int(r["agbd_count"]))
    except Exception as e:  # noqa
        print("   GEDI L4A gagal:", str(e)[:120])
    for a in ESA_CCI_ASSETS:
        try:
            img = ee.ImageCollection(a).sort("system:time_start", False).first()
            img = img.select([img.bandNames().get(0)])
            v = img.reduceRegion(ee.Reducer.mean(), geom, 100, maxPixels=1e9).values().get(0).getInfo()
            if v is not None:
                out["ESA CCI Biomass"] = (float(v), None)
                break
        except Exception as e:  # noqa
            print("   ESA CCI (%s) gagal: %s" % (a, str(e)[:120]))
    try:
        v = ee.Image("LARSE/GEDI/GEDI04_B_002").select("MU").reduceRegion(ee.Reducer.mean(), geom.buffer(500), 1000, maxPixels=1e9).get("MU").getInfo()
        if v is not None:
            out["GEDI L4B (1 km)"] = (float(v), None)
    except Exception as e:  # noqa
        print("   GEDI L4B gagal:", str(e)[:120])
    return out


def severity_ha(geom, start, end):
    region = geom.buffer(J.REF_RING_M[1] + 1500).bounds()
    pre_c = J.collection(region, (start - dt.timedelta(days=J.PRE_DAYS)).isoformat(), start.isoformat())
    post_c = J.collection(region, (end + dt.timedelta(days=J.POST_FROM)).isoformat(), (end + dt.timedelta(days=J.POST_MAX + 1)).isoformat())
    if pre_c.size().getInfo() == 0 or post_c.size().getInfo() == 0:
        return None
    pre, post = J.indices(pre_c.median()), J.indices(post_c.median())
    ring = geom.buffer(4000).difference(geom.buffer(2000), 1)
    d = J.delta_indices(pre, post, ring)
    dn = d["dNBR"].unmask(-1)          # piksel tanpa data (awan) → kelas terendah (konservatif)
    area = ee.Image.pixelArea().divide(1e4)
    bands = [area.updateMask(dn.gte(lo).And(dn.lt(hi))).rename("c%d" % k) for k, (lo, hi, _, _) in enumerate(CC_BY_CLASS)]
    r = ee.Image.cat(bands).reduceRegion(ee.Reducer.sum(), geom, 10, maxPixels=1e10).getInfo()
    return [float(r.get("c%d" % k) or 0) for k in range(len(CC_BY_CLASS))]


def main():
    J.init_ee()
    gwp_key = (os.environ.get("JFEW_GWP") or "AR5").upper()
    gwp = GWP.get(gwp_key, GWP["AR5"])
    fc = json.load(open(os.path.join(os.path.dirname(__file__) or ".", "drone_reference.geojson")))
    results = []
    for f in fc["features"]:
        p = f["properties"]
        geom = ee.Geometry(f["geometry"])
        start, end = dt.date.fromisoformat(p["tgl_api"]), dt.date.fromisoformat(p["tgl_padam"])
        print("== %s · %s · drone %.2f ha" % (p["id"], p["tgl_api"], p["luas_ha"]))
        srcs = agb_sources(geom, start.isoformat())
        for k, v in srcs.items():
            print("   biomassa %-18s %6.1f t/ha%s" % (k, v[0], "" if v[1] is None else " (%d tembakan)" % v[1]))
        for pick in ("GEDI L4A", "ESA CCI Biomass", "GEDI L4B (1 km)"):
            if pick in srcs:
                agb, src = srcs[pick][0], pick
                break
        else:
            agb, src = FALLBACK_AGB, "nilai cadangan %.0f t/ha (ASUMSI)" % FALLBACK_AGB
        sev = severity_ha(geom, start, end)
        tot_px = sum(sev) if sev else 0
        if not sev or tot_px <= 0:
            sev = [0, p["luas_ha"], 0, 0, 0]          # tanpa citra → semua dianggap "rendah"
            tot_px = p["luas_ha"]
        scale = p["luas_ha"] / tot_px                 # samakan dengan luas resmi drone
        sev = [s * scale for s in sev]
        fuel_t = sum(sev[k] * agb * CC_BY_CLASS[k][2] for k in range(len(CC_BY_CLASS)))   # t bahan kering terbakar
        gas = {g: fuel_t * EF_G_PER_KG[g] / 1000.0 for g in EF_G_PER_KG}                   # t gas
        co2e = sum(gas[g] * gwp[g] for g in gas)
        r = {"id": p["id"], "tgl_api": p["tgl_api"], "luas_ha": p["luas_ha"], "agb_t_ha": round(agb, 1), "sumber_biomassa": src,
             "biomassa_semua": {k: round(v[0], 1) for k, v in srcs.items()},
             "ha_kelas": [round(s, 2) for s in sev], "bahan_terbakar_t": round(fuel_t, 1),
             "co2_t": round(gas["CO2"], 1), "ch4_t": round(gas["CH4"], 2), "n2o_t": round(gas["N2O"], 3),
             "co2e_t": round(co2e, 1), "co2e_rendah_t": round(co2e * UNC_LOW, 1), "co2e_tinggi_t": round(co2e * UNC_HIGH, 1),
             "gwp": gwp_key, "metode": "A×B×CC×EF · CC per kelas dNBR · EF IPCC 2006 T2.5 (tropical forest)"}
        print("   kelas dNBR (ha): %s" % ", ".join("%s %.1f" % (CC_BY_CLASS[k][3], sev[k]) for k in range(len(sev))))
        print("   → bahan terbakar %.0f t · CO2 %.0f t · CH4 %.1f t · N2O %.2f t · TOTAL %.0f t CO2e (%s; rentang %.0f–%.0f)"
              % (fuel_t, gas["CO2"], gas["CH4"], gas["N2O"], co2e, gwp_key, co2e * UNC_LOW, co2e * UNC_HIGH))
        results.append(r)
    print(J.web({"action": "carbon_result", "results": results}))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as fh:
            fh.write("## J-FEW · Estimasi emisi karbon kebakaran (boundary drone)\n\n| Kejadian | Luas (ha) | Biomassa (t/ha) · sumber | Bahan terbakar (t) | CO₂e (t, %s) | Rentang (t) |\n|---|---|---|---|---|---|\n" % gwp_key)
            for r in results:
                fh.write("| %s | %.2f | %.1f · %s | %.0f | **%.0f** | %.0f – %.0f |\n" % (r["id"], r["luas_ha"], r["agb_t_ha"], r["sumber_biomassa"], r["bahan_terbakar_t"], r["co2e_t"], r["co2e_rendah_t"], r["co2e_tinggi_t"]))
            fh.write("\n⚠ Estimasi: faktor emisi, GWP, fraksi terbakar (CC) dan biomassa satelit adalah ASUMSI — verifikasi sebelum pelaporan resmi.\n")


if __name__ == "__main__":
    main()
