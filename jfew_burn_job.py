#!/usr/bin/env python3
"""
J-FEW · Luas terbakar aktual otomatis (dNBR Sentinel-2 + Landsat 8/9, Google Earth Engine)
-----------------------------------------------------------------------------------------
1. Ambil antrean event CLOSED dari web app J-FEW  (action "burn_queue")
2. Hitung dNBR per event (metode = JFEW_GEE_Burned_Area_dNBR.js v1.3)
3. Kirim hasil ke web app (action "burn_result") → sheet JFEW_BURNED_AREA, peta, Telegram
Env (GitHub Secrets): EE_SERVICE_ACCOUNT_KEY (JSON), EE_PROJECT (opsional),
                      JFEW_WEBAPP_URL (URL /exec), JFEW_BURN_TOKEN
"""
import datetime as dt
import json
import os
import sys
import urllib.request

import ee

SEARCH_RADIUS_M = 1000
SEED_RADIUS_M = 300
REF_RING_M = (2000, 4000)
DNBR_MIN = 0.20
NDVI_DROP_MIN = 0.08
NDVI_PRE_MIN = 0.25
MIN_PATCH_HA = 0.2
PRE_DAYS = 25
POST_FROM = 2
POST_MAX = 20            # hari setelah padam (atau s.d. hari ini bila lebih awal)
MIN_CLEAR = 0.6          # minimal 60% area pencarian tertutup citra sesudah yang bebas awan
MAX_GEOJSON_CHARS = 45000


def web(payload):
    payload = dict(payload, token=os.environ["JFEW_BURN_TOKEN"].strip())
    req = urllib.request.Request(os.environ["JFEW_WEBAPP_URL"].strip(), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    if not data.get("ok"):
        raise RuntimeError("Web app menolak: %s" % data.get("error"))
    return data


def init_ee():
    key = os.environ["EE_SERVICE_ACCOUNT_KEY"]
    info = json.loads(key)
    creds = ee.ServiceAccountCredentials(info["client_email"], key_data=key)
    ee.Initialize(creds, project=os.environ.get("EE_PROJECT") or info.get("project_id"))


def mask_s2(img):
    scl = img.select("SCL")
    ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return (img.updateMask(ok).select(["B4", "B8", "B12", "B3"], ["red", "nir", "swir2", "green"])
            .divide(10000).toFloat().copyProperties(img, ["system:time_start"]))


def mask_l89(img):
    qa = img.select("QA_PIXEL")
    ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)).And(qa.bitwiseAnd(1 << 1).eq(0))
    sr = img.select(["SR_B4", "SR_B5", "SR_B7", "SR_B3"], ["red", "nir", "swir2", "green"]).multiply(0.0000275).add(-0.2)
    return sr.updateMask(ok).toFloat().copyProperties(img, ["system:time_start"])


def collection(region, d0, d1):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(d0, d1)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)).map(mask_s2))
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
          .filterBounds(region).filterDate(d0, d1).map(mask_l89))
    return s2.merge(l8).map(lambda i: ee.Image(i).toFloat())


def indices(img):
    return (img.addBands(img.normalizedDifference(["nir", "swir2"]).rename("NBR"))
               .addBands(img.normalizedDifference(["nir", "red"]).rename("NDVI"))
               .addBands(img.normalizedDifference(["green", "nir"]).rename("NDWI")))


def rnd(c):
    return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 5), round(c[1], 5)]


def analyse(ev):
    pts = ee.Geometry.MultiPoint(ev["pts"])
    region = pts.buffer(REF_RING_M[1]).bounds()
    search = pts.buffer(SEARCH_RADIUS_M)
    start = dt.date.fromisoformat(ev["start"])
    end = dt.date.fromisoformat(ev["end"])
    post_end = min(end + dt.timedelta(days=POST_MAX), dt.date.today())
    pre_c = collection(region, (start - dt.timedelta(days=PRE_DAYS)).isoformat(), start.isoformat())
    post_c = collection(region, (end + dt.timedelta(days=POST_FROM)).isoformat(), (post_end + dt.timedelta(days=1)).isoformat())
    n_pre, n_post = pre_c.size().getInfo(), post_c.size().getInfo()
    base = {"eventId": ev["id"], "tgl_api": ev["start"], "tgl_padam": ev["end"], "zona": ev.get("zone", ""),
            "citra_sebelum": n_pre, "citra_sesudah": n_post}
    if n_pre == 0 or n_post == 0:
        return dict(base, status="PENDING", catatan="Citra %s belum tersedia" % ("sebelum" if n_pre == 0 else "sesudah"))

    pre, post = indices(pre_c.median()), indices(post_c.median())
    clear = (post.select("NBR").mask().reduceRegion(ee.Reducer.mean(), search, 20, maxPixels=1e9)
             .values().get(0).getInfo() or 0)
    if clear < MIN_CLEAR:
        return dict(base, status="PENDING", tutupan_bersih=round(clear, 2), catatan="Tertutup awan (%d%% bersih)" % (clear * 100))

    dnbr_raw = pre.select("NBR").subtract(post.select("NBR"))
    ring = pts.buffer(REF_RING_M[1]).difference(pts.buffer(REF_RING_M[0]), 1)
    off = (dnbr_raw.updateMask(pre.select("NDVI").gte(NDVI_PRE_MIN))
           .reduceRegion(ee.Reducer.median(), ring, 20, maxPixels=1e9).values().get(0).getInfo() or 0)
    dnbr = dnbr_raw.subtract(off).rename("dNBR")
    ndvi_drop = pre.select("NDVI").subtract(post.select("NDVI"))
    burned = (dnbr.gte(DNBR_MIN).And(ndvi_drop.gte(NDVI_DROP_MIN)).And(pre.select("NDVI").gte(NDVI_PRE_MIN))
              .And(post.select("NDWI").lt(0.1)).clip(search).focalMode(1).selfMask())
    vec = (burned.reduceToVectors(geometry=search, scale=10, geometryType="polygon", eightConnected=True, maxPixels=1e9)
           .filterBounds(pts.buffer(SEED_RADIUS_M))
           .map(lambda p: p.set("ha", p.geometry().area(1).divide(1e4)))
           .filter(ee.Filter.gte("ha", MIN_PATCH_HA)))
    if vec.size().getInfo() == 0:
        return dict(base, status="NOT_DETECTED", luas_ha=0, tutupan_bersih=round(clear, 2), catatan="dNBR offset %.3f" % off)

    geom = vec.geometry(1)
    conn = ee.Image(0).paint(vec, 1).selfMask()
    sev = dnbr.updateMask(conn).expression("b >= 0.66 ? 4 : b >= 0.44 ? 3 : b >= 0.27 ? 2 : 1", {"b": dnbr})
    area = ee.Image.pixelArea().divide(1e4)
    stats = {"luas": geom.area(1).divide(1e4),
             "dnbr": dnbr.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("dNBR")}
    for k in (1, 2, 3, 4):
        stats["s%d" % k] = area.updateMask(sev.eq(k)).reduceRegion(ee.Reducer.sum(), geom, 10, maxPixels=1e9).get("area")
    s = ee.Dictionary(stats).getInfo()
    gj = None
    for tol in (5, 10, 20, 40):
        g = geom.simplify(tol).getInfo()
        g = {"type": g["type"], "coordinates": rnd(g["coordinates"])}
        if len(json.dumps(g, separators=(",", ":"))) <= MAX_GEOJSON_CHARS:
            gj = g
            break
    r2 = lambda v: round(float(v or 0), 2)
    return dict(base, status="DONE", luas_ha=r2(s["luas"]), dnbr_rata2=round(float(s["dnbr"] or 0), 3),
                ha_rendah=r2(s["s1"]), ha_sedang_rendah=r2(s["s2"]), ha_sedang_tinggi=r2(s["s3"]), ha_tinggi=r2(s["s4"]),
                tutupan_bersih=round(clear, 2), catatan="dNBR offset %.3f" % off, geojson=gj)


def main():
    init_ee()
    queue = web({"action": "burn_queue"}).get("events", [])
    print("Antrean: %d event" % len(queue))
    results = []
    for ev in queue:
        try:
            r = analyse(ev)
        except Exception as e:  # noqa
            r = {"eventId": ev["id"], "status": "PENDING", "catatan": "Error: %s" % str(e)[:200]}
        print("  %s → %s %s ha %s" % (ev["id"], r["status"], r.get("luas_ha", "-"), r.get("catatan", "")))
        results.append(r)
    if results:
        print(web({"action": "burn_result", "results": results}))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## J-FEW · Luas terbakar aktual\n\n| Event | Status | Luas (ha) | Catatan |\n|---|---|---|---|\n")
            for r in results:
                f.write("| %s | %s | %s | %s |\n" % (r["eventId"], r["status"], r.get("luas_ha", "-"), r.get("catatan", "")))
            if not results:
                f.write("| – | Tidak ada event yang perlu dianalisis | | |\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa
        print("GAGAL:", e)
        sys.exit(1)
