"""
J-FEW · SIMULASI GENANGAN RAIN-ON-GRID (jfew_flood_sim.py · v1.2 · cari data di beberapa lokasi + pesan jelas · v1.1 · 12 Nov 2026 · data medan boleh di folder utama repo)
Model aliran permukaan 2D "local inertial" (Bates dkk., 2010 — dipakai LISFLOOD-FP), mirip mode diffusion wave HEC-RAS 2D.
Medan: flood_data/dem40.npz (topo EOM Agu 2026, grid ±40 m). Void (EOP 2025) dan sungai = penampung/keluaran.
Limpasan: SCS Curve Number. Dipakai oleh jfew_rain_extreme.py (skenario R5/R25/R100) dan nanti prakiraan harian.
"""
import json
import os
import time

import numpy as np
import shapely
from shapely.geometry import shape, Point
from shapely.ops import transform
from pyproj import Transformer

NEED = ["dem40.npz", "voids.geojson", "rivers.geojson", "catchments_rev06_utm50s.geojson"]
DATA = next((d for d in ["flood_data/", "", "flood_data/flood_data/"] if all(os.path.exists(d + f) for f in NEED)), None)


def load():
    if DATA is None:
        have = {f: [d or "./" for d in ["flood_data/", "", "flood_data/flood_data/"] if os.path.exists(d + f)] for f in NEED}
        raise FileNotFoundError("Data medan belum lengkap di repo. Yang ditemukan: %s · isi folder utama: %s" % (have, sorted(os.listdir("."))[:60]))
    d = np.load(DATA + "dem40.npz"); Z = d["Z"]; x0, y0, dg = float(d["x0"]), float(d["y0"]), float(d["dx"])
    ny, nx = Z.shape
    X, Y = np.meshgrid(x0 + np.arange(nx) * dg, y0 + np.arange(ny) * dg)
    voids = json.load(open(DATA + "voids.geojson"))["features"]
    vid = np.zeros(Z.shape, int)
    for k, f in enumerate(voids, 1):
        vid[shapely.contains_xy(shape(f["geometry"]), X, Y)] = k
    riv = shapely.union_all([shape(f["geometry"]) for f in json.load(open(DATA + "rivers.geojson"))["features"]])
    river = shapely.contains_xy(riv, X, Y)
    T = Transformer.from_crs(32750, 4326, always_xy=True).transform
    cat = [(f["properties"], transform(T, shape(f["geometry"]))) for f in json.load(open(DATA + "catchments_rev06_utm50s.geojson"))["features"]]
    return dict(Z=Z, X=X, Y=Y, x0=x0, y0=y0, dg=dg, voids=voids, vid=vid, river=river, cat=cat)


def scs_excess(hyeto_mm, CN):
    cum = np.cumsum(hyeto_mm); S = 25.4 * (1000 / CN - 10); Ia = 0.2 * S
    Q = np.where(cum > Ia, (cum - Ia) ** 2 / (cum - Ia + S), 0)
    return np.diff(np.r_[0, Q])


def run(G, hyeto_mm, CN=85, n=0.045, extra_h=6, verbose=True):
    """hyeto_mm: hujan per jam (mm). Hasil: genangan maks, kecepatan maks, jam puncak, volume masuk tiap void."""
    Z, vid, river = G["Z"], G["vid"], G["river"]
    DX, g = 40.0, 9.81
    out = np.isnan(Z); z = np.where(out, -50, Z)
    pit = (~out) & (vid == 0) & (z < 0)
    sink = out | (vid > 0) | river | pit
    ex = scs_excess(np.asarray(hyeto_mm, float), CN)
    ny, nx = Z.shape
    h = np.zeros(Z.shape); qx = np.zeros((ny, nx + 1)); qy = np.zeros((ny + 1, nx))
    hmax = np.zeros(Z.shape); vmax = np.zeros(Z.shape); tpk = np.zeros(Z.shape)
    vin = np.zeros(len(G["voids"]) + 1)
    T = (len(hyeto_mm) + extra_h) * 3600; t = 0.0; t0 = time.time()
    while t < T:
        hm = h[~sink].max()
        dt = min(30.0, 0.7 * DX / np.sqrt(g * max(hm, 0.05)))
        eta = z + h
        for ax in (1, 0):
            if ax == 1:
                hf = np.maximum(np.maximum(eta[:, 1:], eta[:, :-1]) - np.maximum(z[:, 1:], z[:, :-1]), 0); qi = qx[:, 1:-1]; s = (eta[:, 1:] - eta[:, :-1]) / DX
            else:
                hf = np.maximum(np.maximum(eta[1:, :], eta[:-1, :]) - np.maximum(z[1:, :], z[:-1, :]), 0); qi = qy[1:-1, :]; s = (eta[1:, :] - eta[:-1, :]) / DX
            qn = (qi - g * hf * dt * s) / (1 + g * dt * n * n * np.abs(qi) / np.maximum(hf, 1e-6) ** (7 / 3))
            qn[hf < 1e-3] = 0; qn = np.clip(qn, -hf * DX / dt / 4, hf * DX / dt / 4)
            if ax == 1: qx[:, 1:-1] = qn
            else: qy[1:-1, :] = qn
        k = int(t // 3600); rain = (ex[k] if k < len(ex) else 0) / 1000 / 3600
        h = np.maximum(h + dt * ((qx[:, :-1] - qx[:, 1:]) + (qy[:-1, :] - qy[1:, :])) / DX + rain * dt, 0)
        vin += np.bincount(vid.ravel(), weights=h.ravel(), minlength=len(vin)) * DX * DX
        h[sink] = 0
        v = np.hypot((qx[:, :-1] + qx[:, 1:]) / 2, (qy[:-1, :] + qy[1:, :]) / 2) / np.maximum(h, 0.05)
        up = h > hmax; tpk[up] = t / 3600; hmax = np.maximum(hmax, h); vmax = np.maximum(vmax, np.where(h > 0.05, np.minimum(v, 6), 0))
        t += dt
        if verbose and int(t // 3600) != int((t - dt) // 3600) and int(t // 3600) % 6 == 0:
            print("   jam %d · dt %.1f s · %.0f s" % (t // 3600, dt, time.time() - t0), flush=True)
    tot = float(np.sum(hyeto_mm))
    for kk in range(1, len(vin)):
        vin[kk] += (vid == kk).sum() * DX * DX * tot / 1000                   # hujan langsung di permukaan void
    return dict(hmax=hmax, vmax=vmax, tpk=tpk, vin=vin, sink=sink)


def summarize(G, R):
    """status tiap void (naik, sisa ke spillway) + luas genangan + titik genangan utama per catchment"""
    from scipy import ndimage
    spill = {p["NAME"].upper().replace(" ", ""): p["SPILL_M"] for p, g in G["cat"]}
    voids = []
    for k, f in enumerate(G["voids"], 1):
        p = f["properties"]; nm = p["nama"].replace("Void ", ""); sp = spill.get(nm.upper()) or spill.get((p.get("alias") or "").upper())
        rise = R["vin"][k] / (p["luas_ha"] * 1e4); new = p["elev_actual"] + rise
        fb = (sp - new) if isinstance(sp, (int, float)) else None
        st = "–" if fb is None else "MELUAP" if fb < 0 else "KRITIS" if fb < 1 else "WASPADA" if fb < 3 else "AMAN"
        voids.append(dict(void=nm, ma_awal=p["elev_actual"], naik_m=round(rise, 2), ma_akhir=round(new, 2), spill=sp,
                          sisa_m=None if fb is None else round(fb, 2), status=st, volume_m3=int(R["vin"][k])))
    land = ~R["sink"]; A = 40 * 40 / 1e4; hm = R["hmax"]
    area = {">0,1 m": round(float((land & (hm > 0.1)).sum() * A), 1), ">0,3 m": round(float((land & (hm > 0.3)).sum() * A), 1),
            ">0,5 m": round(float((land & (hm > 0.5)).sum() * A), 1), ">1 m": round(float((land & (hm > 1)).sum() * A), 1),
            "arus berbahaya": round(float((land & (hm > 0.2) & (hm * R["vmax"] > 0.6)).sum() * A), 1)}
    lab, n = ndimage.label(land & (hm > 0.5)); spots = []
    for i in range(1, n + 1):
        m = lab == i; a = m.sum() * A
        if a < 1.5: continue
        jj, ii = np.nonzero(m); cx = G["x0"] + ii.mean() * G["dg"]; cy = G["y0"] + jj.mean() * G["dg"]
        where = next((p["NAME"] for p, g in G["cat"] if g.contains(Point(cx, cy))), "di luar catchment Rev06")
        spots.append(dict(lokasi=where, luas_ha=round(a, 1), dalam_maks_m=round(float(hm[m].max()), 1), puncak_jam=round(float(np.median(R["tpk"][m]))), lon=round(cx, 5), lat=round(cy, 5)))
    spots.sort(key=lambda s: -s["luas_ha"])
    return dict(voids=voids, area_ha=area, spots=spots[:10])


def plot(G, R, S, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    Z = G["Z"]; ny, nx = Z.shape; dg = G["dg"]
    ex = [G["x0"] - dg / 2, G["x0"] + (nx - .5) * dg, G["y0"] - dg / 2, G["y0"] + (ny - .5) * dg]
    fig, ax = plt.subplots(figsize=(15, 7), dpi=100)
    Zf = np.nan_to_num(Z, nan=np.nanmean(Z)); hs = LightSource(315, 45).hillshade(Zf, vert_exag=4, dx=40, dy=40)
    ax.imshow(np.where(np.isnan(Z), np.nan, hs), cmap="gray", extent=ex, origin="lower", vmin=0, vmax=1.2)
    land = ~R["sink"]; hm = R["hmax"]
    ax.imshow(np.where(land & (hm > 0.1), hm, np.nan), cmap=ListedColormap(["#B3E5FC", "#4FC3F7", "#0288D1", "#01579B"]),
              norm=BoundaryNorm([0.1, 0.3, 0.5, 1, 99], 4), extent=ex, origin="lower", alpha=0.9)
    ax.imshow(np.where(land & (hm > 0.2) & (hm * R["vmax"] > 0.6), 1, np.nan), cmap=ListedColormap(["#E53935"]), extent=ex, origin="lower")
    col = {"MELUAP": "#B71C1C", "KRITIS": "#E65100", "WASPADA": "#F9A825", "AMAN": "#2E7D32", "–": "#546E7A"}
    for f, v in zip(G["voids"], S["voids"]):
        g = shape(f["geometry"])
        for gg in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            a = np.array(gg.exterior.coords); ax.fill(a[:, 0], a[:, 1], color="#1A237E", alpha=.55, ec=col[v["status"]], lw=2)
        x, y = g.representative_point().coords[0]
        ax.text(x, y, "%s +%.1f m\n%s" % (v["void"], v["naik_m"], v["status"]), fontsize=7.5, ha="center", va="center", color="white",
                bbox=dict(fc=col[v["status"]], ec="none", alpha=.9, pad=1.2))
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3]); ax.set_aspect(1 / np.cos(np.radians(3.9))); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[Patch(fc=c, label=l) for c, l in [("#B3E5FC", "10–30 cm"), ("#4FC3F7", "30–50 cm"), ("#0288D1", "0,5–1 m"), ("#01579B", "> 1 m"),
              ("#E53935", "arus berbahaya"), ("#B71C1C", "void meluap"), ("#E65100", "void kritis"), ("#F9A825", "void waspada"), ("#2E7D32", "void aman")]],
              loc="lower left", fontsize=8, ncol=3, framealpha=.9)
    plt.tight_layout(); plt.savefig(path); plt.close(fig)
