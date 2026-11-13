"""
J-FEW · PETA PERINGATAN TSUNAMI untuk Telegram (jfew_tsunami_map.py · v1 · 13 Nov 2026)
Dipicu dari Apps Script (ZZZ_JFEW_TSUNAMI.gs) lewat workflow_dispatch saat alert tsunami terkirim.
Menggambar: daratan, zona perairan yang terhubung ke pantai JBG, radius 300 km & 1.500 km,
port JBG, pusat gempa + garis jarak → PNG → dikirim ke web app (action "tsunami_img"),
lalu Apps Script meneruskannya ke Telegram sebagai foto.
Butuh: matplotlib · data daratan: jfew_land_indo.json (repo root)
"""
import base64
import io
import json
import math
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Polygon                            # noqa: E402

PORT = (114.970, -4.027)
NEAR_KM, MAX_KM = 300, 1500
ZONA = [[105.0, -2.6], [105.5, -5.9], [108.0, -6.2], [111.0, -6.4], [114.5, -6.9], [115.6, -8.0],
        [117.0, -8.3], [120.0, -8.3], [123.5, -8.1], [123.5, -5.0], [120.5, -5.0], [119.2, -3.0],
        [119.0, 0.0], [119.5, 1.6], [117.8, 1.6], [116.3, -1.5], [116.0, -3.6], [114.6, -3.6],
        [113.0, -3.1], [110.5, -2.8], [108.5, -1.5], [106.0, -1.8]]


def env(k, d=""):
    return (os.environ.get(k) or d).strip()


def circle(lon, lat, km, n=180):
    """lingkaran geodesik (titik tujuan pada jarak km dari pusat)"""
    R, d = 6371.0, km / 6371.0
    la1, lo1 = math.radians(lat), math.radians(lon)
    out = []
    for i in range(n + 1):
        b = 2 * math.pi * i / n
        la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
        lo2 = lo1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
        out.append((math.degrees(lo2), math.degrees(la2)))
    return out


def merc(lat):
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def main():
    lat, lon = float(env("Q_LAT", "-5.2")), float(env("Q_LON", "112.8"))
    mag, km = env("Q_MAG", "7.1"), env("Q_KM", "274")
    wil, pot = env("Q_WILAYAH", "-"), env("Q_POTENSI", "Berpotensi tsunami")
    waktu, uji = env("Q_WAKTU", ""), env("Q_UJI", "0") == "1"

    here = os.path.dirname(os.path.abspath(__file__))
    land_f = next(p for p in (os.path.join(here, "jfew_land_indo.json"), "jfew_land_indo.json", "burn_auto/jfew_land_indo.json") if os.path.exists(p))
    land = json.load(open(land_f))

    # batas peta: zona + pusat gempa, sedikit ruang
    xs = [p[0] for p in ZONA] + [lon, PORT[0]]
    ys = [p[1] for p in ZONA] + [lat, PORT[1]]
    x0, x1, y0, y1 = min(xs) - 1.2, max(xs) + 1.2, min(ys) - 1.6, max(ys) + 1.0
    T = lambda pts: [(x, merc(y)) for x, y in pts]          # noqa: E731  (proyeksi Mercator sederhana)

    fig, ax = plt.subplots(figsize=(10, 7.6), dpi=110)
    ax.set_facecolor("#EAF2F8")
    for r in land:
        ax.add_patch(Polygon(T(r), closed=True, fc="#E4E1D6", ec="#B4B2A9", lw=0.5, zorder=1))
    ax.add_patch(Polygon(T(ZONA), closed=True, fc="#1D9E75", alpha=0.22, ec="#0F6E56", lw=1.4, zorder=2))
    c = T(circle(*PORT, NEAR_KM)); ax.fill(*zip(*c), color="#E24B4A", alpha=0.08, zorder=2); ax.plot(*zip(*c), color="#E24B4A", lw=1.5, zorder=3)
    c = T(circle(*PORT, MAX_KM)); ax.plot(*zip(*c), color="#888780", lw=1, ls=(0, (5, 4)), zorder=3)

    # garis jarak & titik
    px, py = PORT[0], merc(PORT[1]); qx, qy = lon, merc(lat)
    ax.plot([qx, px], [qy, py], color="#A32D2D", lw=1.6, ls=(0, (4, 3)), zorder=4)
    ax.text((qx + px) / 2, (qy + py) / 2, f" {km} km ", fontsize=11, color="#A32D2D", weight="bold", ha="center", va="center",
            bbox=dict(fc="white", ec="#A32D2D", boxstyle="round,pad=0.25"), zorder=6)
    for rr, a in ((0.55, 0.15), (0.35, 0.25)):
        ax.add_patch(plt.Circle((qx, qy), rr, color="#C62828", alpha=a, zorder=4))
    ax.plot(qx, qy, marker="*", ms=24, color="#C62828", mec="white", mew=1.5, zorder=7)
    ax.plot(px, py, "o", ms=11, color="#7A1F1F", mec="white", mew=2, zorder=7)
    ax.text(px + 0.25, py + 0.28, "Port / jetty JBG", fontsize=11, weight="bold", zorder=7)
    ax.text(qx + 0.4, qy - 0.45, f"Pusat gempa M {mag.replace('.', ',')}", fontsize=11, weight="bold", color="#C62828", zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

    for t, x, y in (("Laut Jawa", 110.6, -4.9), ("Selat Makassar", 118.1, -1.2), ("Laut Flores", 121.0, -6.6), ("Samudra Hindia", 107.5, -10.4)):
        if x0 < x < x1 and y0 < y < y1:
            ax.text(x, merc(y), t, fontsize=10, style="italic", color="#888780" if t == "Samudra Hindia" else "#0F6E56", ha="center")
    for t, x, y in (("KALIMANTAN", 113.2, -1.0), ("JAWA", 109.6, -7.3), ("SULAWESI", 121.0, -2.2)):
        if x0 < x < x1 and y0 < y < y1:
            ax.text(x, merc(y), t, fontsize=9, color="#777", ha="center", alpha=0.9)

    ax.set_xlim(x0, x1); ax.set_ylim(merc(y0), merc(y1)); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#CCC")
    judul = ("UJI COBA — " if uji else "") + "J-FEW · PERINGATAN TSUNAMI"
    ax.set_title(judul, fontsize=15, weight="bold", color="#A32D2D", loc="left", pad=26)
    ax.text(0, 1.012, f"{wil} · {waktu} · BMKG: {pot}", transform=ax.transAxes, fontsize=9.5, color="#444")
    ax.text(0.01, 0.015, "Hijau = perairan terhubung ke pantai JBG · lingkaran merah 300 km · putus-putus 1.500 km\n"
            "Sumber: BMKG InaTEWS · peta: J-FEW PT JBG · selalu ikuti arahan resmi BMKG/BPBD",
            transform=ax.transAxes, fontsize=8.5, color="#555", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15); png = buf.getvalue()
    open("tsunami_map.png", "wb").write(png)
    print("PNG", len(png), "byte")

    if not env("JFEW_WEBAPP_URL"):
        print("JFEW_WEBAPP_URL kosong — hanya simpan file lokal."); return
    cap = ("🧪 UJI COBA · " if uji else "") + f"🌊 Peta peringatan tsunami · M {mag.replace('.', ',')} · {km} km dari port JBG\n{wil}"
    body = {"action": "tsunami_img", "token": env("JFEW_BURN_TOKEN"), "png": base64.b64encode(png).decode(), "caption": cap}
    req = urllib.request.Request(env("JFEW_WEBAPP_URL"), data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read().decode("utf-8", "replace"))
    print(res)
    if not res.get("ok"):
        raise SystemExit("Web app menolak: %s" % res.get("error"))


if __name__ == "__main__":
    main()
