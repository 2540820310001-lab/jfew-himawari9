"""
J-FEW · STATUS VOID & PRAKIRAAN GENANGAN HARIAN (jfew_void_daily.py · v1.1 · 13 Nov 2026 · genangan disaring, titik genangan utama untuk popup)
Tiap pagi:
  1. muka air & debit pompa terbaru tiap void dari sheet "Muka Air Void" (Apps Script action void_input; cadangan voids.geojson)
  2. prakiraan hujan per jam 72 jam ke depan (Open-Meteo, rata-rata 3 titik area tambang)
  3. simulasi genangan rain-on-grid (jfew_flood_sim.py, topo EOM Agu 2026) → kenaikan tiap void − pompa → sisa ke spillway
  4. kirim ke peta (Apps Script action void_status): status void + citra genangan (PNG)
Mingguan (Senin) atau bila diminta (RUN_SCENARIOS=1): juga skenario R5/R25/R100 dari analisis hujan ekstrem.
Belum mengirim Telegram (sesuai keputusan 13 Nov 2026).
Env: EE tidak diperlukan · JFEW_WEBAPP_URL, JFEW_BURN_TOKEN · opsional RUN_SCENARIOS, CN (85), R_SCN (JSON {"R5":[R1,R3],...})
"""
import base64
import datetime as dt
import io
import json
import os
import urllib.request

import numpy as np

import jfew_burn_job as J
import jfew_flood_sim as M

WITA = dt.timezone(dt.timedelta(hours=8))
PTS = [(115.0, -3.89), (115.04, -3.89), (115.075, -3.885)]
# hujan rencana (IMERG titik ombrometer terkoreksi, jfew_rain_extreme.py run 13 Nov 2026): [hari terbesar, total 3 hari] mm
R_SCN = {"R5": [171, 264], "R25": [224, 318], "R100": [265, 352]}
R2_3D = 214


def get_json(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "J-FEW void"}), timeout=90).read().decode())


def forecast_72h():
    lat = ",".join(str(p[1]) for p in PTS); lon = ",".join(str(p[0]) for p in PTS)
    d = get_json("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&hourly=precipitation&timezone=Asia%%2FMakassar&forecast_days=4" % (lat, lon))
    d = d if isinstance(d, list) else [d]
    t = d[0]["hourly"]["time"]; P = np.nanmean([np.array(x["hourly"]["precipitation"], float) for x in d], axis=0)
    now = dt.datetime.now(WITA).strftime("%Y-%m-%dT%H:00")
    i0 = next((i for i, s in enumerate(t) if s >= now), 0)
    hy = np.nan_to_num(P[i0:i0 + 72]); return hy, t[i0], t[min(i0 + 71, len(t) - 1)]


def design_hyeto(R1, R3):
    t = np.arange(1, 25); depth = R1 / 24 * (24 / t) ** (2 / 3) * t; inc = np.diff(np.r_[0, depth])
    order = [3]; l, r = 2, 4
    while len(order) < 24:
        if r < 24: order.append(r); r += 1
        if l >= 0 and len(order) < 24: order.append(l); l -= 1
    d2 = np.zeros(24); d2[order] = inc; rest = max(R3 - R1, 0) / 48
    return np.r_[np.full(24, rest), d2, np.full(24, rest)]


def levels(G):
    """muka air & pompa terbaru: sheet → cadangan voids.geojson"""
    base = {f["properties"]["nama"].replace("Void ", ""): dict(ma=f["properties"]["elev_actual"], tgl=f["properties"].get("tgl_ukur"), pompa=0, sumber="data void J-FEW")
            for f in G["voids"]}
    try:
        r = J.web({"action": "void_input"}) or {}
        for k, v in (r.get("voids") or {}).items():
            k2 = "M2EE" if k.upper() in ("M23EE", "M2EE") else k
            if k2 in base and v.get("ma") is not None:
                base[k2].update(ma=float(v["ma"]), tgl=v.get("tgl"), pompa=float(v.get("pompa") or 0), sumber="sheet Muka Air Void")
    except Exception as ex:                                                    # noqa: BLE001
        print("void_input gagal, pakai data bawaan:", ex)
    return base


def png(G, R):
    """citra genangan RGBA (sel 40 m) untuk L.imageOverlay"""
    from PIL import Image
    hm = np.where(M.wet_mask(R), R["hmax"], 0); ny, nx = hm.shape          # v1.1: tanpa sel genangan tunggal yang dangkal
    rgba = np.zeros((ny, nx, 4), np.uint8)
    for lo, hi, c in [(0.1, 0.3, (179, 229, 252, 170)), (0.3, 0.5, (79, 195, 247, 190)), (0.5, 1, (2, 136, 209, 210)), (1, 999, (1, 87, 155, 225))]:
        rgba[(hm >= lo) & (hm < hi)] = c
    rgba[(hm > 0.2) & (hm * R["vmax"] > 0.6)] = (229, 57, 53, 230)
    im = Image.fromarray(rgba[::-1]); b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def status(sisa):
    return "MELUAP" if sisa < 0 else "KRITIS" if sisa < 1 else "WASPADA" if sisa < 3 else "AMAN"


def one(G, L, hy, label, CN, pump_hours):
    R = M.run(G, hy, CN=CN, verbose=False); S = M.summarize(G, R)
    out = []
    for f, v in zip(G["voids"], S["voids"]):
        nm = v["void"]; lv = L[nm]; A = f["properties"]["luas_ha"] * 1e4
        pump = lv["pompa"] * pump_hours / A
        rise = max(v["naik_m"] - pump, -pump); ma1 = lv["ma"] + rise
        sisa = None if not isinstance(v["spill"], (int, float)) else round(v["spill"] - ma1, 2)
        out.append(dict(void="M23EE" if nm == "M2EE" else nm, ma=lv["ma"], tgl=lv["tgl"], sumber=lv["sumber"], pompa=lv["pompa"], naik=round(rise, 2), ma_pred=round(ma1, 2),
                        spill=v["spill"], sisa=sisa, status="–" if sisa is None else status(sisa), volume_m3=v["volume_m3"]))
    return dict(label=label, hujan_mm=round(float(np.sum(hy)), 1), hujan_maks_jam=round(float(np.max(hy)), 1), voids=out,
                area_ha=S["area_ha"], spots=S["spots"][:10], png=png(G, R))


def main():
    G = M.load(); L = levels(G); CN = float(os.environ.get("CN") or 85)
    hy, t0, t1 = forecast_72h()
    print("prakiraan hujan 72 jam: %.1f mm (maks %.1f mm/jam) %s → %s" % (hy.sum(), hy.max(), t0, t1), flush=True)
    res = {"issued": dt.datetime.now(WITA).strftime("%Y-%m-%d %H:%M"), "cn": CN, "r2_3d": R2_3D,
           "bounds": [[G["y0"] - G["dg"] / 2, G["x0"] - G["dg"] / 2], [G["y0"] + (G["Z"].shape[0] - .5) * G["dg"], G["x0"] + (G["Z"].shape[1] - .5) * G["dg"]]],
           "geom": [{"void": {"M2EE": "M23EE"}.get(f["properties"]["nama"].replace("Void ", ""), f["properties"]["nama"].replace("Void ", "")), "geometry": f["geometry"]} for f in G["voids"]]}
    fc = one(G, L, hy, "Prakiraan 72 jam (%s – %s)" % (t0.replace("T", " "), t1.replace("T", " ")), CN, 72)
    fc["hyeto"] = [round(float(v), 1) for v in hy]; res["forecast"] = fc
    run_scn = os.environ.get("RUN_SCENARIOS") == "1" or dt.datetime.now(WITA).weekday() == 0
    if run_scn:
        rs = json.loads(os.environ["R_SCN"]) if os.environ.get("R_SCN") else R_SCN
        res["scenarios"] = {}
        for k, (r1, r3) in rs.items():
            print("skenario", k, flush=True)
            res["scenarios"][k] = one(G, L, design_hyeto(r1, r3), "%s · hujan 3 hari %d mm (hari terbesar %d mm)" % (k, r3, r1), CN, 72)
    print(J.web({"action": "void_status", "result": res}))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        md = ["# J-FEW · status void & genangan", "", "%s · hujan %.0f mm (maks %.0f mm/jam) · R2 3 hari = %d mm" % (fc["label"], fc["hujan_mm"], fc["hujan_maks_jam"], R2_3D), "",
              "| Void | MA (tgl, sumber) | Pompa m³/jam | Naik | MA prakiraan | Spillway | Sisa | Status |", "|---|---|---|---|---|---|---|---|"]
        md += ["| %s | %s (%s, %s) | %s | %+.2f | %.2f | %s | %s | %s |" % (v["void"], v["ma"], v["tgl"], v["sumber"], v["pompa"], v["naik"], v["ma_pred"], v["spill"], v["sisa"], v["status"]) for v in fc["voids"]]
        for k, s in (res.get("scenarios") or {}).items():
            md += ["", "**%s**: " % s["label"] + " · ".join("%s %s" % (v["void"], v["status"]) for v in s["voids"] if v["status"] != "AMAN")]
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("\n".join(md))


if __name__ == "__main__":
    main()
