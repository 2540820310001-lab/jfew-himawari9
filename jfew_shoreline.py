"""
J-FEW · PERUBAHAN GARIS PANTAI (jfew_shoreline.py · v1.1 · 13 Nov 2026 · NaN/inf dibersihkan sebelum dikirim · Summary ditulis sebelum kirim)
Metode ala DSAS (Digital Shoreline Analysis System, USGS) di pantai konsesi PT JBG:
  1. Komposit tahunan Landsat 5/7/8/9 Collection-2 SR (semua citra bebas awan dalam 1 tahun, median)
     → MNDWI = (Hijau − SWIR1)/(Hijau + SWIR1); air bila MNDWI > 0. Median setahun meredam efek pasang surut.
  2. Laut = komponen air yang tersambung ke tepi selatan/timur kotak (kolam, void, sungai pedalaman dibuang).
  3. Garis dasar (baseline) = garis pantai tahun terakhir yang dihaluskan; transek tegak lurus tiap 100 m,
     panjang 1,5 km (750 m ke darat, 750 m ke laut).
  4. Posisi garis pantai tiap tahun di tiap transek → laju LRR (regresi linier, m/tahun, selang 95 %),
     EPR (titik akhir), laju 10 tahun terakhir; pencilan dibuang (3×MAD).
  5. Proyeksi 10 & 20 tahun (linier, dengan selang kepercayaan) + jarak garis pantai ke aset (port, jalan, dll).
Kirim ke peta: action "shore".  Jalan: GitHub Actions setahun sekali (Januari) atau manual.
Butuh: earthengine-api numpy scipy scikit-image
"""
import datetime as dt
import json
import math
import os
import traceback

import numpy as np
from scipy import ndimage, stats

import jfew_burn_job as J

# ---------------- konfigurasi ----------------
BOX = (114.86, -4.18, 115.22, -3.86)          # W, S, E, N — pantai Jorong s.d. Asam-Asam
SEG_LON = (114.88, 115.20)                     # transek hanya di rentang bujur ini
PIX_DEG = 0.00027                              # ≈ 30 m
YEAR0 = int(os.environ.get("SHORE_YEAR0") or 1990)
STEP_M, HALF_M, SAMPLE_M = 100, 750, 10
MIN_YEARS = 10                                 # minimal tahun valid per transek
STABLE = 1.0                                   # |laju| < 1 m/th dianggap stabil
PROJ = (10, 20)                                # proyeksi 10 & 20 tahun
PORT = (114.970, -4.027)
# Aset pesisir yang dipantau jaraknya ke garis pantai — tambahkan titik lain di sini (nama, bujur, lintang)
ASSETS = [("Port / jetty JBG", 114.970, -4.027)]
try:
    ASSETS += [tuple(a) for a in json.loads(os.environ.get("SHORE_ASSETS") or "[]")]
except Exception:                                                     # noqa: BLE001
    pass

W0, S0, E0, N0 = BOX
NX, NY = int(round((E0 - W0) / PIX_DEG)), int(round((N0 - S0) / PIX_DEG))
LAT0 = (S0 + N0) / 2
MX, MY = 111320 * math.cos(math.radians(LAT0)) * PIX_DEG, 110574 * PIX_DEG     # meter per piksel (x, y)


# ---------------- Earth Engine ----------------
def landsat_year(y):
    import ee
    reg = ee.Geometry.Rectangle(list(BOX))
    d0, d1 = "%d-01-01" % y, "%d-01-01" % (y + 1)

    def prep(col, g, s):
        def f(im):
            qa = im.select("QA_PIXEL")
            ok = qa.bitwiseAnd(0b11111).eq(0)                   # fill, dilated cloud, cirrus, cloud, shadow
            sr = im.select([g, s]).multiply(0.0000275).add(-0.2)
            return sr.normalizedDifference([g, s]).rename("m").updateMask(ok)
        return ee.ImageCollection(col).filterBounds(reg).filterDate(d0, d1).filter(ee.Filter.lt("CLOUD_COVER", 80)).map(f)
    c = prep("LANDSAT/LT05/C02/T1_L2", "SR_B2", "SR_B5").merge(prep("LANDSAT/LE07/C02/T1_L2", "SR_B2", "SR_B5")) \
        .merge(prep("LANDSAT/LC08/C02/T1_L2", "SR_B3", "SR_B6")).merge(prep("LANDSAT/LC09/C02/T1_L2", "SR_B3", "SR_B6"))
    n = c.size()
    img = ee.Image(ee.Algorithms.If(n.gt(0), c.median(), ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename("m")))
    return img.multiply(100).round().clamp(-100, 100).unmask(-128).toInt8().rename("y%d" % y), n


def fetch(years):
    import ee
    out = {}
    grid = {"dimensions": {"width": NX, "height": NY},
            "affineTransform": {"scaleX": PIX_DEG, "shearX": 0, "translateX": W0, "shearY": 0, "scaleY": -PIX_DEG, "translateY": N0},
            "crsCode": "EPSG:4326"}
    for i in range(0, len(years), 6):
        ys = years[i:i + 6]
        ims = [landsat_year(y) for y in ys]
        stack = ee.Image.cat([a for a, _ in ims])
        arr = ee.data.computePixels({"expression": stack, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        cnt = ee.List([b for _, b in ims]).getInfo()
        for y, k in zip(ys, cnt):
            a = np.array(arr["y%d" % y]).astype(np.int16)
            out[y] = (a, k)
        print("tahun", ys, "citra", cnt)
    return out


# ---------------- analisis (numpy murni, bisa diuji tanpa GEE) ----------------
def sea_mask(m):
    """m: int MNDWI×100, −128 = tanpa data → (laut bool, valid bool)"""
    valid = m != -128
    water = (m > 0) & valid
    lab, n = ndimage.label(water)
    if n == 0:
        return np.zeros_like(water), valid
    edge = np.unique(np.concatenate([lab[-1, :], lab[:, -1]]))     # tepi selatan & timur
    edge = edge[edge > 0]
    if not len(edge):
        return np.zeros_like(water), valid
    sizes = ndimage.sum(water, lab, edge)
    keep = edge[sizes >= max(sizes.max() * 0.2, 500)]
    return np.isin(lab, keep), valid


def px2ll(c, r):
    return W0 + (c + 0.5) * PIX_DEG, N0 - (r + 0.5) * PIX_DEG


def ll2px(lon, lat):
    return (lon - W0) / PIX_DEG - 0.5, (N0 - lat) / PIX_DEG - 0.5


def baseline(sea):
    """garis pantai dihaluskan (≈300 m) → titik tiap STEP_M + normal ke arah darat; dalam koordinat meter-piksel"""
    from skimage import measure
    sm = ndimage.gaussian_filter(sea.astype(float), 10)
    cs = measure.find_contours(sm, 0.5)
    if not cs:
        raise RuntimeError("garis pantai tidak ditemukan di kotak BOX")
    c = max(cs, key=len)                                       # (row, col)
    P = np.c_[c[:, 1] * MX, c[:, 0] * MY]                       # meter (x ke timur, y ke selatan)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = np.arange(0, seg[-1], STEP_M)
    x, yv = np.interp(s, seg, P[:, 0]), np.interp(s, seg, P[:, 1])
    tx, ty = np.gradient(x), np.gradient(yv); L = np.hypot(tx, ty); tx, ty = tx / L, ty / L
    nx, ny = -ty, tx                                            # normal
    out = []
    for xi, yi, a, b in zip(x, yv, nx, ny):
        # arahkan normal ke darat: cek sampel 200 m di sisi +normal
        cc, rr = int(round((xi + a * 200) / MX)), int(round((yi + b * 200) / MY))
        if 0 <= rr < sea.shape[0] and 0 <= cc < sea.shape[1] and sea[rr, cc]:
            a, b = -a, -b
        lon, lat = px2ll(xi / MX, yi / MY)
        if SEG_LON[0] <= lon <= SEG_LON[1]:
            out.append((xi, yi, a, b))                          # (a,b) = arah ke darat
    return out


def shore_pos(sea, valid, T):
    """posisi garis pantai (m dari ujung darat, positif ke laut) atau None"""
    xi, yi, a, b = T
    d = np.arange(-HALF_M, HALF_M + 1, SAMPLE_M)               # dari darat (−) ke laut (+) relatif baseline
    xs, ys = xi - a * d, yi - b * d                             # −a: menjauhi darat
    cc, rr = np.round(xs / MX).astype(int), np.round(ys / MY).astype(int)
    if cc.min() < 0 or rr.min() < 0 or cc.max() >= sea.shape[1] or rr.max() >= sea.shape[0]:
        return None
    sv, vv = sea[rr, cc], valid[rr, cc]
    if not sv[-1]:
        return None
    i = len(d) - 1
    while i >= 0:
        if not vv[i]:
            return None
        if not sv[i] and not sv[max(0, i - 3):i + 1].any():
            return float(d[i] + HALF_M + SAMPLE_M / 2)           # jarak dari ujung darat transek
        i -= 1
    return None


def regress(yrs, pos):
    yrs, pos = np.asarray(yrs, float), np.asarray(pos, float)
    if len(yrs) < MIN_YEARS:
        return None
    for _ in range(2):                                          # buang pencilan 3×MAD
        sl, ic = np.polyfit(yrs, pos, 1)
        res = pos - (sl * yrs + ic); mad = np.median(np.abs(res - np.median(res))) * 1.4826 or 1
        k = np.abs(res) <= 3 * mad
        if k.all() or k.sum() < MIN_YEARS:
            break
        yrs, pos = yrs[k], pos[k]
    if np.ptp(pos) == 0:                                        # posisi sama persis tiap tahun → stabil (laju 0); R² tidak terdefinisi
        n = len(yrs); xm = yrs.mean()
        return dict(slope=0.0, ic=float(pos[0]), ci=0.0, r2=float("nan"), n=n, yrs=yrs, pos=pos, tc=stats.t.ppf(0.975, n - 2), se=0.0, xm=xm, sxx=((yrs - xm) ** 2).sum())
    r = stats.linregress(yrs, pos)
    n = len(yrs); tc = stats.t.ppf(0.975, n - 2)
    if not all(np.isfinite([r.slope, r.intercept, r.stderr, r.rvalue])):
        return None
    xm = yrs.mean(); sxx = ((yrs - xm) ** 2).sum(); se = math.sqrt(max(((pos - (r.slope * yrs + r.intercept)) ** 2).sum() / (n - 2), 0))
    return dict(slope=r.slope, ic=r.intercept, ci=tc * r.stderr, r2=r.rvalue ** 2, n=n, yrs=yrs, pos=pos, tc=tc, se=se, xm=xm, sxx=sxx)


def predict(R, year):
    v = R["slope"] * year + R["ic"]
    half = R["tc"] * R["se"] * math.sqrt(1 + 1 / R["n"] + (year - R["xm"]) ** 2 / R["sxx"])
    return v, half


def analyze(stack, years):
    last = max(y for y in years if stack[y][1] > 0)
    sea_last, _ = sea_mask(stack[last][0])
    TS = baseline(sea_last)
    masks = {y: sea_mask(stack[y][0]) for y in years if stack[y][1] > 0}
    now = dt.date.today().year
    out = []
    for i, T in enumerate(TS):
        xi, yi, a, b = T
        lon, lat = px2ll((xi - a * -HALF_M) / MX, (yi - b * -HALF_M) / MY)     # ujung darat
        lon1, lat1 = px2ll((xi - a * HALF_M) / MX, (yi - b * HALF_M) / MY)     # ujung laut
        L = 2 * HALF_M
        pos = {y: shore_pos(masks[y][0], masks[y][1], T) for y in masks}
        pv = [(y, p) for y, p in pos.items() if p is not None]
        R = regress([y for y, _ in pv], [p for _, p in pv]) if pv else None
        tr = dict(id=i + 1, o=[round(lon, 5), round(lat, 5)], e=[round(lon1, 5), round(lat1, 5)], len=L,
                  pos=[None if pos.get(y) is None else int(round(pos[y])) for y in years])
        if R:
            ok = set(R["yrs"].astype(int).tolist())
            r10 = [(y, p) for y, p in pv if y >= now - 11 and y in ok]
            l10 = np.polyfit([y for y, _ in r10], [p for _, p in r10], 1)[0] if len(r10) >= 6 else None
            first, lastp = min(pv), max(pv)
            tr.update(lrr=round(float(R["slope"]), 2), ci=round(float(R["ci"]), 2), r2=round(float(R["r2"]), 2), n=int(R["n"]),
                      epr=round(float((lastp[1] - first[1]) / max(1, lastp[0] - first[0])), 2),
                      lrr10=None if l10 is None else round(float(l10), 2),
                      fit=[round(float(R["slope"]), 4), round(float(R["ic"]), 1)],
                      proj={str(now + k): [round(float(v)), round(float(h))] for k in PROJ for v, h in [predict(R, now + k)]},
                      now=round(float(predict(R, now)[0])))
            sig = abs(R["slope"]) > R["ci"]
            tr["cls"] = "stabil" if (abs(R["slope"]) < STABLE or not sig) else ("abrasi" if R["slope"] < 0 else "akresi")
        else:
            tr["cls"] = "data kurang"
        out.append(tr)
    return out, last


def at(tr, p):
    """titik pada transek di jarak p (m dari ujung darat) → [lon, lat]"""
    f = p / tr["len"]
    return [tr["o"][0] + (tr["e"][0] - tr["o"][0]) * f, tr["o"][1] + (tr["e"][1] - tr["o"][1]) * f]


def km_ll(a, b):
    return math.hypot((a[0] - b[0]) * 111.32 * math.cos(math.radians(a[1])), (a[1] - b[1]) * 110.574)


def arah(dx, dy):
    k = ["utara", "timur laut", "timur", "tenggara", "selatan", "barat daya", "barat", "barat laut"]
    return k[int(((math.degrees(math.atan2(dx, dy)) + 360 + 22.5) % 360) // 45)]


def describe(pt):
    dx = (pt[0] - PORT[0]) * 111.32 * math.cos(math.radians(PORT[1])); dy = (pt[1] - PORT[1]) * 110.574
    d = math.hypot(dx, dy)
    return "di sekitar port" if d < 0.3 else "±%s km %s port" % (str(round(d, 1)).replace(".", ","), arah(dx, dy))


def assets(trs):
    now = dt.date.today().year; out = []
    for name, lon, lat in ASSETS:
        best = None
        for t in trs:
            if "fit" not in t:
                continue
            # proyeksi aset ke sumbu transek
            vx = (t["e"][0] - t["o"][0]) * 111320 * math.cos(math.radians(lat)); vy = (t["e"][1] - t["o"][1]) * 110574
            wx = (lon - t["o"][0]) * 111320 * math.cos(math.radians(lat)); wy = (lat - t["o"][1]) * 110574
            L = math.hypot(vx, vy); along = (wx * vx + wy * vy) / L; off = abs(wx * vy - wy * vx) / L
            if off > 150 or along < -500 or along > L + 500:
                continue
            if best is None or off < best[1]:
                best = (t, off, along)
        if not best:
            out.append(dict(nama=name, lon=lon, lat=lat, tid=None)); continue
        t, off, along = best
        jarak = t["now"] - along                                 # m dari aset ke garis pantai (positif = pantai di depan aset)
        sl = t["lrr"]; tahun0 = None
        if sl < -0.05 and jarak > 0:
            tahun0 = int(now + jarak / -sl)
        out.append(dict(nama=name, lon=lon, lat=lat, tid=t["id"], jarak=int(round(jarak)), laju=float(sl), tahun_nol=tahun0,
                        proj={k: int(round(v[0] - along)) for k, v in t["proj"].items()}))
    return out


def summary(trs, ast):
    v = [t for t in trs if "lrr" in t and t["cls"] != "data kurang"]
    w = min(v, key=lambda t: t["lrr"]) if v else None
    b = max(v, key=lambda t: t["lrr"]) if v else None
    cnt = {k: sum(1 for t in trs if t["cls"] == k) for k in ("abrasi", "akresi", "stabil", "data kurang")}
    S = dict(n=len(trs), kelas=cnt)
    for k, t in (("terparah", w), ("terbesar", b)):
        if t:
            p = at(t, t["now"]); S[k] = dict(id=t["id"], laju=float(t["lrr"]), ci=float(t["ci"]), lon=round(p[0], 5), lat=round(p[1], 5), lokasi=describe(p))
    return S


def clean(o):
    """NaN / inf → None (JSON standar), numpy → tipe Python"""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def main():
    now = dt.date.today()
    last_full = now.year - 1 if now.month < 7 else now.year      # tahun berjalan dipakai bila sudah ≥ 6 bulan
    years = list(range(YEAR0, last_full + 1))
    J.init_ee()
    stack = fetch(years)
    trs, last = analyze(stack, years)
    ast = assets(trs)
    res = dict(issued=now.isoformat(), years=years, last=last, box=BOX, step=STEP_M, transects=trs, assets=ast,
               summary=summary(trs, ast), proj_years=[now.year + k for k in PROJ], scenes={str(y): stack[y][1] for y in years},
               metode="Landsat 5/7/8/9 C2 SR, komposit median tahunan, MNDWI>0, transek %d m, LRR (DSAS) selang 95%%" % STEP_M)
    res = clean(res)
    s = json.dumps(res, allow_nan=False)
    print("transek", len(trs), "· ukuran", len(s) // 1024, "KB")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        S = res["summary"]; ast = res["assets"]
        md = ["# J-FEW · perubahan garis pantai", "", "Tahun %d–%d · %d transek · kelas %s" % (years[0], last, len(trs), S["kelas"]), ""]
        if "terparah" in S:
            md.append("- Abrasi terparah: %s ± %s m/th (%s)" % (S["terparah"]["laju"], S["terparah"]["ci"], S["terparah"]["lokasi"]))
        if "terbesar" in S:
            md.append("- Akresi terbesar: %s ± %s m/th (%s)" % (S["terbesar"]["laju"], S["terbesar"]["ci"], S["terbesar"]["lokasi"]))
        for a in ast:
            md.append("- %s: jarak ke garis pantai %s m, laju %s m/th%s" % (a["nama"], a.get("jarak"), a.get("laju"), ", pantai mencapai aset ± %s" % a["tahun_nol"] if a.get("tahun_nol") else ""))
        md += ["", "Jumlah citra per tahun: " + ", ".join("%s:%s" % (y, stack[y][1]) for y in years)]
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md) + "\n\n")
    print(J.web({"action": "shore", "result": res}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("# J-FEW garis pantai — GAGAL\n\n```\n%s\n```\n" % traceback.format_exc()[-3000:])
        raise
