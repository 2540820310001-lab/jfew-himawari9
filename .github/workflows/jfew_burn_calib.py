#!/usr/bin/env python3
"""
J-FEW · Kalibrasi indeks area terbakar terhadap boundary DRONE (jfew_burn_calib.py)
-----------------------------------------------------------------------------------
Membandingkan dNBR, dNBR2, dMIRBI dan kombinasinya dengan boundary lapangan (drone_reference.geojson).
Untuk tiap metode/ambang dihitung (dalam radius 1,5 km dari boundary drone):
  luas terdeteksi, luas yang cocok dengan drone, PRECISION (berapa % deteksi benar),
  RECALL (berapa % luas drone tertangkap), F1 dan rasio luas (deteksi / drone).
Hasil: tabel di ringkasan GitHub Actions + file calib_result.csv (artefak).
Tidak mengubah data J-FEW. Env: EE_SERVICE_ACCOUNT_KEY, EE_PROJECT (opsional).
"""
import csv
import datetime as dt
import json
import os

import ee

import jfew_burn_job as J

SEARCH_M = 1500
GRID = {
    "dNBR": [0.10, 0.15, 0.20, 0.27],
    "dNBR2": [0.02, 0.04, 0.06, 0.08, 0.10],
    "dMIRBI": [0.10, 0.20, 0.30, 0.50, 0.80],
}
COMBOS = [(0.20, c, n2, mb) for c in (0.05, 0.10, 0.15) for n2 in (0.03, 0.05, 0.08) for mb in (0.20, 0.30, 0.50)]


def stats(mask, drone, search):
    a = ee.Image.pixelArea().divide(1e4)
    m = mask.unmask(0).clip(search)
    dr = ee.Image(0).paint(drone, 1).clip(search)
    img = ee.Image.cat([a.updateMask(m.eq(1)).rename("det"), a.updateMask(m.eq(1).And(dr.eq(1))).rename("tp"), a.updateMask(dr.eq(1)).rename("ref")])
    r = img.reduceRegion(ee.Reducer.sum(), search, 10, maxPixels=1e10).getInfo()
    det, tp, ref = r.get("det") or 0, r.get("tp") or 0, r.get("ref") or 0
    p = tp / det if det else 0
    rc = tp / ref if ref else 0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0
    return det, tp, ref, p, rc, f1


def main():
    J.init_ee()
    fc = json.load(open(os.path.join(os.path.dirname(__file__) or ".", "drone_reference.geojson")))
    rows = []
    for f in fc["features"]:
        pr = f["properties"]
        drone = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"]))])
        search = drone.geometry().buffer(SEARCH_M)
        region = drone.geometry().buffer(J.REF_RING_M[1] + SEARCH_M).bounds()
        start, end = dt.date.fromisoformat(pr["tgl_api"]), dt.date.fromisoformat(pr["tgl_padam"])
        pre_c = J.collection(region, (start - dt.timedelta(days=J.PRE_DAYS)).isoformat(), start.isoformat())
        post_c = J.collection(region, (end + dt.timedelta(days=J.POST_FROM)).isoformat(), (end + dt.timedelta(days=J.POST_MAX + 1)).isoformat())
        pre, post = J.indices(pre_c.median()), J.indices(post_c.median())
        ring = drone.geometry().buffer(SEARCH_M + 2500).difference(drone.geometry().buffer(SEARCH_M + 500), 1)
        d = J.delta_indices(pre, post, ring)
        base = pre.select("NDVI").gte(J.NDVI_PRE_MIN).And(post.select("NDWI").lt(0.1))
        print("== %s (drone %.2f ha) · citra sebelum %d / sesudah %d · offset dNBR %.3f dNBR2 %.3f dMIRBI %.3f"
              % (pr["id"], pr["luas_ha"], pre_c.size().getInfo(), post_c.size().getInfo(), d["off"], d["off2"], d["offm"]))
        tests = [("%s ≥ %.2f" % (k, t), d[k].gte(t).And(base)) for k in GRID for t in GRID[k]]
        tests.append(("dNBR ≥ 0.20 + NDVI turun (metode v1)", J.burn_masks(d, pre, post)[0]))
        for (s, c, n2, mb) in COMBOS:
            tests.append(("KOMBINASI dNBR≥%.2f | (dNBR≥%.2f & (dNBR2≥%.2f | dMIRBI≥%.2f))" % (s, c, n2, mb),
                          J.burn_masks(d, pre, post, s, c, n2, mb)[1]))
        for name, mask in tests:
            det, tp, ref, p, rc, f1 = stats(mask.focalMode(1), drone, search)
            rows.append([pr["id"], name, round(det, 2), round(tp, 2), round(ref, 2), round(p, 3), round(rc, 3), round(f1, 3), round(det / ref, 2) if ref else 0])
            print("  %-70s deteksi %7.2f ha · cocok %7.2f · P %.2f · R %.2f · F1 %.3f" % (name, det, tp, p, rc, f1))
    head = ["kejadian", "metode", "deteksi_ha", "cocok_ha", "drone_ha", "precision", "recall", "f1", "rasio_luas"]
    with open("calib_result.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(rows)
    # rata-rata F1 per metode → peringkat
    agg = {}
    for r in rows:
        agg.setdefault(r[1], []).append(r)
    rank = sorted(agg.items(), key=lambda kv: -sum(x[7] for x in kv[1]) / len(kv[1]))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as fh:
            fh.write("## J-FEW · Kalibrasi indeks area terbakar vs boundary drone\n\n")
            fh.write("| # | Metode | F1 rata-rata | Recall rata-rata | Precision rata-rata | Rasio luas rata-rata |\n|---|---|---|---|---|---|\n")
            for i, (m, rs) in enumerate(rank[:15], 1):
                n = len(rs)
                fh.write("| %d | %s | %.3f | %.2f | %.2f | %.2f |\n" % (i, m, sum(x[7] for x in rs) / n, sum(x[6] for x in rs) / n, sum(x[5] for x in rs) / n, sum(x[8] for x in rs) / n))
            fh.write("\nRecall = % luas drone yang tertangkap · Precision = % deteksi yang benar · Rasio luas = deteksi / drone. Detail: artefak calib_result.csv\n")
    print("\nPERINGKAT (F1 rata-rata):")
    for i, (m, rs) in enumerate(rank[:10], 1):
        print("%2d. %-70s F1 %.3f" % (i, m, sum(x[7] for x in rs) / len(rs)))


if __name__ == "__main__":
    main()
