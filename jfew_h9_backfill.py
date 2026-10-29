#!/usr/bin/env python3
"""
J-FEW Himawari-9 BACKFILL (cross-check) - versi GitHub Actions
-------------------------------------------------------------
Menelusuri SEMUA file Himawari-9 L2 Wildfire (WLF, tiap 10 menit) di FTP JAXA
P-Tree untuk rentang tanggal START..END (UTC), lalu menyimpan HANYA deteksi
di sekitar konsesi PT JBG (default radius 10 km dari boundary zona Z01-Z04).

Tidak mengunggah ke Drive (supaya folder WLF harian tidak tercampur);
hasil disimpan sebagai artefak GitHub Actions:
  h9_hits_<START>_<END>.csv     -> deteksi dalam radius (lengkap dg baris asli)
  h9_scan_<START>_<END>.csv     -> statistik per hari: folder/file terbaca,
                                   jumlah hotspot Kalimantan, hit konsesi
Env: PTREE_USER, PTREE_PASSWORD, START (YYYY-MM-DD), END (YYYY-MM-DD),
     RADIUS_KM (10), THREADS (3), REMOTE_TEMPLATE (opsional), OUT_DIR (out)
"""
import csv, ftplib, io, json, math, os, re, sys, threading, time, queue
from datetime import datetime, timedelta, timezone

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa
    pass

FILE_PATTERN = re.compile(r"^H09_(\d{8})_(\d{4})_.*L2WLF.*\.csv$", re.IGNORECASE)
HOST = os.environ.get("FTP_HOST") or "ftp.ptree.jaxa.jp"
USER = os.environ.get("PTREE_USER", "")
PASS = os.environ.get("PTREE_PASSWORD", "").strip()
TEMPLATE = (os.environ.get("REMOTE_TEMPLATE") or "/pub/himawari/L2/WLF/010/{yyyymm}/{dd}/{hh}/").strip()
START = os.environ.get("START") or "2026-07-01"
END = os.environ.get("END") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
RADIUS_KM = float(os.environ.get("RADIUS_KM") or "10")
THREADS = int(os.environ.get("THREADS") or "3")
OUT_DIR = os.environ.get("OUT_DIR") or "out"

# Zona cluster PT JBG (disederhanakan ~30 m dari Cluster_*_Jfew.geojson), [lon, lat]
ZONES = json.loads('''{"Z01 - west":[[115.03009,-3.91565],[115.02308,-3.91564],[115.02308,-3.91418],[115.0219,-3.91418],[115.0219,-3.9132],[115.02084,-3.9132],[115.02085,-3.91266],[115.01512,-3.91264],[115.01512,-3.91326],[115.01152,-3.9137],[115.01151,-3.91471],[115.00971,-3.91471],[115.00971,-3.9153],[114.99615,-3.91529],[114.99383,-3.91592],[114.98707,-3.9159],[114.98707,-3.91666],[114.98082,-3.91659],[114.98083,-3.91729],[114.97691,-3.91728],[114.97691,-3.91798],[114.97247,-3.91797],[114.97247,-3.91926],[114.96362,-3.91939],[114.96365,-3.9086],[114.96922,-3.9086],[114.96922,-3.90786],[114.97372,-3.9079],[114.97372,-3.90702],[114.97824,-3.90705],[114.97825,-3.90643],[114.98273,-3.90645],[114.98273,-3.90586],[114.98723,-3.9059],[114.98723,-3.90529],[114.99173,-3.90532],[114.99173,-3.90481],[114.99284,-3.90481],[114.99285,-3.90246],[114.99557,-3.90246],[114.99557,-3.9009],[115.00243,-3.90091],[115.00266,-3.88681],[115.0056,-3.89083],[115.00408,-3.89076],[115.00404,-3.89137],[115.00561,-3.89112],[115.00587,-3.89177],[115.00604,-3.89282],[115.00414,-3.8942],[115.00458,-3.89494],[115.00526,-3.89476],[115.00494,-3.8942],[115.00591,-3.89443],[115.0058,-3.89544],[115.00394,-3.89606],[115.00426,-3.89737],[115.00528,-3.89744],[115.00487,-3.89871],[115.00611,-3.89847],[115.0069,-3.8997],[115.00751,-3.899],[115.00842,-3.89962],[115.00889,-3.89883],[115.0104,-3.8989],[115.0101,-3.89842],[115.01066,-3.89817],[115.01113,-3.89838],[115.01113,-3.89969],[115.01186,-3.89942],[115.01287,-3.90065],[115.01317,-3.90011],[115.01263,-3.89911],[115.01358,-3.89889],[115.01408,-3.90011],[115.01485,-3.90052],[115.01489,-3.90198],[115.01789,-3.90247],[115.01875,-3.90302],[115.0191,-3.90832],[115.01964,-3.91024],[115.02267,-3.9108],[115.02566,-3.91296],[115.0301,-3.91523],[115.03009,-3.91565]],"Z02 - central":[[115.05834,-3.90444],[115.05205,-3.90442],[115.05205,-3.90611],[115.04844,-3.9061],[115.04844,-3.90791],[115.04498,-3.9079],[115.04499,-3.90685],[115.04409,-3.90684],[115.0441,-3.90557],[115.04238,-3.90557],[115.04237,-3.90684],[115.04109,-3.90684],[115.04108,-3.9081],[115.03765,-3.90809],[115.03765,-3.9095],[115.03355,-3.90948],[115.03354,-3.91037],[115.03011,-3.91036],[115.0301,-3.91523],[115.02566,-3.91296],[115.02267,-3.9108],[115.01964,-3.91024],[115.0191,-3.90832],[115.01875,-3.90302],[115.01789,-3.90247],[115.01489,-3.90198],[115.01485,-3.90052],[115.01408,-3.90011],[115.01358,-3.89889],[115.01263,-3.89911],[115.01317,-3.90011],[115.01287,-3.90065],[115.01186,-3.89942],[115.01113,-3.89969],[115.01113,-3.89838],[115.01066,-3.89817],[115.0101,-3.89842],[115.0104,-3.8989],[115.00889,-3.89883],[115.00842,-3.89962],[115.00751,-3.899],[115.0069,-3.8997],[115.00611,-3.89847],[115.00487,-3.89871],[115.00528,-3.89744],[115.00426,-3.89737],[115.00394,-3.89606],[115.0058,-3.89544],[115.00591,-3.89443],[115.00494,-3.8942],[115.00526,-3.89476],[115.00458,-3.89494],[115.00414,-3.8942],[115.00604,-3.89282],[115.00587,-3.89177],[115.00561,-3.89112],[115.00404,-3.89137],[115.00408,-3.89076],[115.0056,-3.89083],[115.00266,-3.88681],[115.00978,-3.88682],[115.00978,-3.88629],[115.01428,-3.8863],[115.01428,-3.88582],[115.01698,-3.88582],[115.01699,-3.88476],[115.02106,-3.88479],[115.02106,-3.88388],[115.02329,-3.88389],[115.02329,-3.88305],[115.02516,-3.88306],[115.02516,-3.88188],[115.0278,-3.88189],[115.0278,-3.88008],[115.0323,-3.88008],[115.0323,-3.8789],[115.03773,-3.87893],[115.03773,-3.88125],[115.03864,-3.88125],[115.03863,-3.88278],[115.03936,-3.88278],[115.03936,-3.88428],[115.04063,-3.88573],[115.04033,-3.88643],[115.03921,-3.887],[115.03925,-3.88749],[115.04608,-3.8937],[115.04904,-3.89463],[115.05238,-3.89708],[115.05595,-3.90336],[115.05801,-3.90378],[115.05834,-3.90444]],"Z03 - east":[[115.06375,-3.90445],[115.0585,-3.90444],[115.05801,-3.90378],[115.05595,-3.90336],[115.05238,-3.89708],[115.04904,-3.89463],[115.04608,-3.8937],[115.03925,-3.88749],[115.03921,-3.887],[115.04033,-3.88643],[115.04063,-3.88573],[115.03936,-3.88428],[115.04282,-3.88429],[115.04283,-3.88126],[115.04467,-3.88127],[115.04465,-3.8801],[115.04693,-3.88011],[115.04693,-3.87671],[115.06045,-3.87673],[115.06045,-3.87552],[115.07408,-3.87557],[115.0741,-3.87306],[115.07589,-3.87307],[115.07589,-3.86997],[115.08129,-3.87],[115.0813,-3.86782],[115.09514,-3.86784],[115.09517,-3.87831],[115.09202,-3.87831],[115.09202,-3.88015],[115.08753,-3.88014],[115.08752,-3.88193],[115.08657,-3.88194],[115.08657,-3.88375],[115.08563,-3.88375],[115.08563,-3.88556],[115.08469,-3.88555],[115.08468,-3.88811],[115.08385,-3.88811],[115.08384,-3.89353],[115.08023,-3.89352],[115.08023,-3.89442],[115.07728,-3.89442],[115.07728,-3.89551],[115.07447,-3.8955],[115.07446,-3.89973],[115.07125,-3.89972],[115.07125,-3.90062],[115.06675,-3.90061],[115.06675,-3.90163],[115.06376,-3.90163],[115.06375,-3.90445]],"Z04 - hauling":[[114.99615,-3.91529],[114.99614,-3.93313],[114.99091,-3.93314],[114.99091,-3.94358],[114.98627,-3.94359],[114.98626,-3.95737],[114.98011,-3.95738],[114.98011,-3.97012],[114.97245,-3.97012],[114.97244,-3.98584],[114.97614,-3.98584],[114.97614,-4.02086],[114.96536,-4.02083],[114.96537,-3.95738],[114.97244,-3.95738],[114.97245,-3.94358],[114.98011,-3.94358],[114.98011,-3.93761],[114.98535,-3.93763],[114.98537,-3.93256],[114.98807,-3.93257],[114.98809,-3.92475],[114.99176,-3.92476],[114.99176,-3.92205],[114.99248,-3.92205],[114.99248,-3.92089],[114.99382,-3.92089],[114.99383,-3.91592],[114.99615,-3.91529]]}''')

# Kotak cepat: Kalimantan Selatan (untuk statistik "ada hotspot di sekitar?")
KALSEL = (114.0, -4.6, 116.6, -1.3)   # W, S, E, N

ALL_PTS = [p for ring in ZONES.values() for p in ring]
BB = (min(p[0] for p in ALL_PTS), min(p[1] for p in ALL_PTS), max(p[0] for p in ALL_PTS), max(p[1] for p in ALL_PTS))
PAD = RADIUS_KM / 110.0 + 0.01


def in_ring(lon, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]; xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def dist_km(lon, lat, ring):
    k = math.cos(math.radians(lat)); best = 1e9
    for a in range(len(ring) - 1):
        x1, y1 = (ring[a][0] - lon) * k, ring[a][1] - lat
        x2, y2 = (ring[a + 1][0] - lon) * k, ring[a + 1][1] - lat
        dx, dy = x2 - x1, y2 - y1; L = dx * dx + dy * dy
        t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / L)) if L else 0.0
        px, py = x1 + t * dx, y1 + t * dy
        best = min(best, math.hypot(px, py) * 111.2)
    return best


def classify(lon, lat):
    for name, ring in ZONES.items():
        if in_ring(lon, lat, ring):
            return name, 0.0
    best = min(((dist_km(lon, lat, r), n) for n, r in ZONES.items()))
    return best[1], best[0]


# ---------- parsing CSV WLF (deteksi kolom lat/lon otomatis) ----------
def parse(content):
    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return [], None
    header, start = None, 0
    for i, l in enumerate(lines[:20]):
        low = l.lower()
        if "lat" in low and "lon" in low:
            header = [h.strip().lstrip("#").strip() for h in l.split(",")]; start = i + 1; break
    rows = []
    ilat = ilon = None
    if header:
        lowh = [h.lower() for h in header]
        ilat = next((i for i, h in enumerate(lowh) if h.startswith("lat")), None)
        ilon = next((i for i, h in enumerate(lowh) if h.startswith("lon")), None)
    for l in lines[start:]:
        if l.lstrip().startswith("#"):
            continue
        v = [x.strip() for x in l.split(",")]
        if ilat is None or ilon is None:      # heuristik: pasangan lat(-90..90), lon(90..200) berurutan
            for i in range(len(v) - 1):
                try:
                    a, b = float(v[i]), float(v[i + 1])
                except ValueError:
                    continue
                if -60 <= a <= 60 and 70 <= b <= 220 and ("." in v[i]) and ("." in v[i + 1]):
                    ilat, ilon = i, i + 1; break
            if ilat is None:
                continue
        try:
            rows.append((float(v[ilat]), float(v[ilon]), v))
        except (ValueError, IndexError):
            continue
    return rows, header


# ---------- FTP ----------
def connect():
    for attempt in range(1, 6):
        try:
            f = ftplib.FTP(timeout=90); f.connect(HOST, 21); f.login(USER, PASS); f.set_pasv(True); return f
        except Exception as e:  # noqa
            print("  FTP connect %d/5 gagal: %s" % (attempt, e)); time.sleep(5 * attempt)
    raise RuntimeError("FTP tidak bisa login")


def main():
    if not USER or not PASS:
        raise SystemExit("PTREE_USER / PTREE_PASSWORD belum diisi (Secrets).")
    os.makedirs(OUT_DIR, exist_ok=True)
    d0 = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(END, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    d1 = min(d1, datetime.now(timezone.utc))
    print("Rentang UTC %s .. %s | radius %.0f km | bbox zona %s" % (d0.date(), (d1 - timedelta(seconds=1)).date(), RADIUS_KM, BB))

    jobs = queue.Queue()
    t = d0
    while t < d1:
        jobs.put(t); t += timedelta(hours=1)
    total_hours = jobs.qsize()

    lock = threading.Lock()
    hits, header_seen = [], [None]
    stats = {}   # yyyy-mm-dd -> dict
    done = [0]

    def day_stat(day):
        return stats.setdefault(day, {"hours_ok": 0, "hours_missing": 0, "files": 0, "files_failed": 0,
                                      "hotspots_disk": 0, "hotspots_kalsel": 0, "hits_radius": 0, "hits_inside": 0})

    def worker():
        ftp = connect()
        while True:
            try:
                h = jobs.get_nowait()
            except queue.Empty:
                break
            day = h.strftime("%Y-%m-%d")
            path = TEMPLATE.format(yyyy=h.strftime("%Y"), mm=h.strftime("%m"), yyyymm=h.strftime("%Y%m"),
                                   dd=h.strftime("%d"), hh=h.strftime("%H"), yyyymmdd=h.strftime("%Y%m%d"))
            names = None
            for attempt in range(3):
                try:
                    names = [n.rsplit("/", 1)[-1] for n in ftp.nlst(path)]; break
                except ftplib.error_perm:
                    names = []; break
                except Exception:  # noqa
                    try: ftp.close()
                    except Exception: pass  # noqa
                    ftp = connect()
            names = sorted(n for n in (names or []) if FILE_PATTERN.match(n))
            local = {"files": 0, "failed": 0, "disk": 0, "kalsel": 0, "hits": []}
            for n in names:
                buf = io.BytesIO(); ok = False
                for attempt in range(3):
                    try:
                        ftp.retrbinary("RETR " + path.rstrip("/") + "/" + n, buf.write); ok = True; break
                    except Exception:  # noqa
                        buf = io.BytesIO()
                        try: ftp.close()
                        except Exception: pass  # noqa
                        ftp = connect()
                if not ok:
                    local["failed"] += 1; continue
                rows, header = parse(buf.getvalue().decode("utf-8", "replace"))
                if header and header_seen[0] is None:
                    header_seen[0] = header
                local["files"] += 1; local["disk"] += len(rows)
                m = FILE_PATTERN.match(n)
                obs = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                for lat, lon, raw in rows:
                    if KALSEL[0] <= lon <= KALSEL[2] and KALSEL[1] <= lat <= KALSEL[3]:
                        local["kalsel"] += 1
                    if BB[0] - PAD <= lon <= BB[2] + PAD and BB[1] - PAD <= lat <= BB[3] + PAD:
                        zone, dkm = classify(lon, lat)
                        if dkm <= RADIUS_KM:
                            local["hits"].append({
                                "obs_utc": obs.strftime("%Y-%m-%d %H:%M"),
                                "obs_wita": (obs + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
                                "lat": lat, "lon": lon,
                                "status": "DALAM KONSESI" if dkm == 0 else "LUAR (%.1f km)" % dkm,
                                "zona_terdekat": zone, "jarak_km": round(dkm, 2),
                                "file": n, "baris_asli": ",".join(raw)})
            with lock:
                s = day_stat(day)
                s["hours_ok" if names else "hours_missing"] += 1
                s["files"] += local["files"]; s["files_failed"] += local["failed"]
                s["hotspots_disk"] += local["disk"]; s["hotspots_kalsel"] += local["kalsel"]
                s["hits_radius"] += len(local["hits"])
                s["hits_inside"] += sum(1 for x in local["hits"] if x["jarak_km"] == 0)
                hits.extend(local["hits"])
                done[0] += 1
                if done[0] % 24 == 0 or done[0] == total_hours:
                    print("  %d/%d jam | file %d | hit radius %d" % (done[0], total_hours,
                          sum(v["files"] for v in stats.values()), len(hits)))
        try: ftp.quit()
        except Exception: pass  # noqa

    ths = [threading.Thread(target=worker) for _ in range(max(1, THREADS))]
    [x.start() for x in ths]; [x.join() for x in ths]

    tag = "%s_%s" % (START, END)
    hits.sort(key=lambda x: x["obs_utc"])
    with open(os.path.join(OUT_DIR, "h9_hits_%s.csv" % tag), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["obs_utc", "obs_wita", "lat", "lon", "status", "zona_terdekat",
                                          "jarak_km", "file", "baris_asli"])
        w.writeheader(); w.writerows(hits)
    with open(os.path.join(OUT_DIR, "h9_scan_%s.csv" % tag), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tanggal_utc", "jam_ada_folder", "jam_tanpa_folder", "file_wlf", "file_gagal",
                    "hotspot_full_disk", "hotspot_kalsel", "hit_radius_%dkm" % RADIUS_KM, "hit_dalam_konsesi"])
        for day in sorted(stats):
            s = stats[day]
            w.writerow([day, s["hours_ok"], s["hours_missing"], s["files"], s["files_failed"],
                        s["hotspots_disk"], s["hotspots_kalsel"], s["hits_radius"], s["hits_inside"]])
    summary = {
        "range": [START, END], "files": sum(v["files"] for v in stats.values()),
        "days_without_files": [d for d, v in sorted(stats.items()) if v["files"] == 0],
        "hotspots_kalsel": sum(v["hotspots_kalsel"] for v in stats.values()),
        "hits_radius": len(hits), "hits_inside": sum(1 for x in hits if x["jarak_km"] == 0),
        "csv_header": header_seen[0]}
    with open(os.path.join(OUT_DIR, "h9_summary_%s.json" % tag), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
