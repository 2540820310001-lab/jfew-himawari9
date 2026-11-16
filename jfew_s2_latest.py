"""
J-FEW · CITRA SENTINEL-2 TERBARU TANPA AWAN (jfew_s2_latest.py · v1 · 16 Nov 2026)
Komposit "piksel bersih terbaru" untuk area konsesi PT JBG + sekitarnya:
  1. Semua citra Sentinel-2 L2A (COPERNICUS/S2_SR_HARMONIZED) LOOKBACK hari terakhir.
  2. Awan & bayangan awan dibuang per piksel dengan Google Cloud Score+ (cs_cdf ≥ CS_MIN = bersih).
  3. Citra diurutkan lama → baru lalu di-mosaic: piksel bersih TERBARU di atas, bagian berawan
     terisi otomatis oleh citra bersih sebelumnya.
  4. Hasil: (a) warna asli B4-B3-B2, (b) warna semu bekas terbakar B12-B8-B4 (SWIR),
     (c) peta umur piksel (hari sejak akuisisi) + statistik.
  5. Gambar (JPEG/PNG) dikirim ke web app J-FEW (action "s2_img" & "s2_meta") → disimpan di Google Drive
     → tampil sebagai layer peta.
Butuh: earthengine-api numpy pillow · Jalan: GitHub Actions tiap Senin + manual.
"""
import base64
import datetime as dt
import io
import json
import os
import traceback

import numpy as np
from PIL import Image

import jfew_burn_job as J

BOX = (114.90, -4.07, 115.17, -3.80)                  # W, S, E, N — konsesi + ±2 km, port & pantai
PIX = float(os.environ.get("S2_PIX_DEG") or 0.0001)   # ≈ 11 m
LOOKBACK = int(os.environ.get("S2_LOOKBACK") or 90)    # hari
CS_MIN = float(os.environ.get("S2_CS_MIN") or 0.60)    # ambang Cloud Score+ (0,5–0,65 umum dipakai)
TILE = 900                                             # ukuran potongan unduhan (piksel)
WITA = dt.timezone(dt.timedelta(hours=8))

W0, S0, E0, N0 = BOX
NX, NY = int(round((E0 - W0) / PIX)), int(round((N0 - S0) / PIX))


def composite(today):
    import ee
    reg = ee.Geometry.Rectangle(list(BOX))
    d1 = ee.Date(today.isoformat()).advance(1, "day")
    d0 = d1.advance(-LOOKBACK, "day")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(reg).filterDate(d0, d1) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 95))
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = s2.linkCollection(cs, ["cs_cdf"])
    epoch = ee.Date("2020-01-01")

    def prep(im):
        ok = im.select("cs_cdf").gte(CS_MIN)
        day = ee.Image.constant(ee.Date(im.get("system:time_start")).difference(epoch, "day").floor()).toInt16().rename("day")
        return im.select(["B2", "B3", "B4", "B8", "B12"]).addBands(day).updateMask(ok) \
                 .set("system:time_start", im.get("system:time_start"))
    col = s2.map(prep).sort("system:time_start")       # lama → baru; mosaic: yang terakhir (terbaru) di atas
    mos = col.mosaic()
    info = s2.aggregate_array("system:time_start").getInfo(), s2.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    rgb = mos.visualize(bands=["B4", "B3", "B2"], min=0, max=2800, gamma=1.25)
    swir = mos.visualize(bands=["B12", "B8", "B4"], min=[0, 0, 0], max=[3500, 4500, 2800], gamma=1.15)
    day = mos.select("day").unmask(-1).toInt16()
    stack = rgb.rename(["r", "g", "b"]).addBands(swir.rename(["s1", "s2", "s3"])).addBands(day).toInt16()   # satu tipe data untuk computePixels
    return stack, info, epoch


def fetch(stack):
    import ee
    out = {k: np.zeros((NY, NX), np.int16 if k == "day" else np.uint8) for k in ["r", "g", "b", "s1", "s2", "s3", "day"]}
    out["day"][:] = -1
    for r0 in range(0, NY, TILE):
        for c0 in range(0, NX, TILE):
            h, w = min(TILE, NY - r0), min(TILE, NX - c0)
            grid = {"dimensions": {"width": w, "height": h},
                    "affineTransform": {"scaleX": PIX, "shearX": 0, "translateX": W0 + c0 * PIX,
                                        "shearY": 0, "scaleY": -PIX, "translateY": N0 - r0 * PIX},
                    "crsCode": "EPSG:4326"}
            a = ee.data.computePixels({"expression": stack, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            for k in out:
                out[k][r0:r0 + h, c0:c0 + w] = np.array(a[k]).astype(out[k].dtype)
        print("baris", r0, "/", NY)
    return out


def enc(img, fmt, **kw):
    b = io.BytesIO(); img.save(b, fmt, **kw); return b.getvalue()


def main():
    today = dt.datetime.now(WITA).date()
    J.init_ee()
    stack, (times, clouds), _ = composite(today)
    if not times:
        raise SystemExit("Tidak ada citra Sentinel-2 dalam %d hari terakhir." % LOOKBACK)
    A = fetch(stack)
    epoch = dt.date(2020, 1, 1)
    day = A["day"].astype(np.int32)
    have = day >= 0
    age = np.where(have, (today - epoch).days - day, -1)

    gray = np.array([150, 150, 150], np.uint8)
    rgb = np.dstack([A["r"], A["g"], A["b"]]); rgb[~have] = gray
    sw = np.dstack([A["s1"], A["s2"], A["s3"]]); sw[~have] = gray
    j_rgb = enc(Image.fromarray(rgb), "JPEG", quality=82, optimize=True, progressive=True)
    j_sw = enc(Image.fromarray(sw), "JPEG", quality=80, optimize=True, progressive=True)

    # peta umur piksel (RGBA): ≤7 hijau · 8–30 kuning · 31–60 oranye · >60 merah · tanpa data abu-abu
    pal = np.zeros(age.shape + (4,), np.uint8)
    for cond, col in ((age < 0, (120, 120, 120, 150)), ((age >= 0) & (age <= 7), (46, 125, 50, 120)),
                      ((age > 7) & (age <= 30), (249, 168, 37, 130)), ((age > 30) & (age <= 60), (230, 81, 0, 140)),
                      (age > 60, (183, 28, 28, 150))):
        pal[cond] = col
    small = Image.fromarray(pal).resize((NX // 3, NY // 3), Image.NEAREST)
    p_age = enc(small, "PNG", optimize=True)

    # grid umur kecil untuk klik di peta (±220 m/sel, 1 byte/sel, 255 = tanpa data)
    g = age[::20, ::20].copy(); g = np.where(g < 0, 255, np.clip(g, 0, 254)).astype(np.uint8)
    n = have.size
    stats = dict(le7=round(float(((age >= 0) & (age <= 7)).sum()) / n * 100, 1),
                 le30=round(float(((age >= 0) & (age <= 30)).sum()) / n * 100, 1),
                 gt30=round(float((age > 30).sum()) / n * 100, 1),
                 none=round(float((~have).sum()) / n * 100, 1))
    used = sorted(set(int(v) for v in np.unique(day[have])))
    scenes = sorted({(dt.datetime.utcfromtimestamp(t / 1000).date().isoformat(), round(c, 1)) for t, c in zip(times, clouds)})
    meta = dict(issued=dt.datetime.now(WITA).strftime("%Y-%m-%d %H:%M"), box=BOX, lookback=LOOKBACK, cs_min=CS_MIN,
                newest=(epoch + dt.timedelta(days=max(used))).isoformat() if used else None,
                oldest=(epoch + dt.timedelta(days=min(used))).isoformat() if used else None,
                dates_used=[(epoch + dt.timedelta(days=d)).isoformat() for d in used],
                scenes=scenes[-40:], stats=stats, size=[NX, NY], pix_m=round(PIX * 111320),
                age_grid=dict(w=g.shape[1], h=g.shape[0], b64=base64.b64encode(g.tobytes()).decode(), today=today.isoformat()))
    print("ukuran: warna asli %d KB · SWIR %d KB · umur %d KB" % (len(j_rgb) // 1024, len(j_sw) // 1024, len(p_age) // 1024))
    print("statistik", stats, "· tanggal dipakai", meta["dates_used"])

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        md = ["# J-FEW · citra Sentinel-2 terbaru (tanpa awan)", "",
              "Komposit %s s.d. %s · %d tanggal citra dipakai · resolusi ±%d m" % (meta["oldest"], meta["newest"], len(used), meta["pix_m"]), "",
              "| Umur piksel | Porsi area |", "|---|---|",
              "| ≤ 7 hari | %s %% |" % stats["le7"], "| ≤ 30 hari | %s %% |" % stats["le30"],
              "| > 30 hari | %s %% |" % stats["gt30"], "| tanpa data (berawan terus) | %s %% |" % stats["none"], "",
              "Citra tersedia (tanggal · awan %): " + ", ".join("%s · %s" % s for s in scenes[-15:])]
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md) + "\n\n")

    for kind, mime, data in (("rgb", "image/jpeg", j_rgb), ("swir", "image/jpeg", j_sw), ("age", "image/png", p_age)):
        print(kind, J.web({"action": "s2_img", "kind": kind, "mime": mime, "b64": base64.b64encode(data).decode()}))
    print(J.web({"action": "s2_meta", "meta": meta}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("# J-FEW citra Sentinel-2 — GAGAL\n\n```\n%s\n```\n" % traceback.format_exc()[-3000:])
        raise
