# -*- coding: utf-8 -*-
"""
J-FEW Himawari-9 WLF Downloader - versi GitHub Actions
======================================================
Dijalankan oleh GitHub Actions setiap 10 menit.

Alur:
  1. Tanya Receiver (Apps Script) file apa saja yang sudah ada di Drive.
  2. Login FTP JAXA P-Tree, cek folder WLF untuk LOOKBACK_HOURS terakhir.
  3. File yang belum ada diunduh lalu dikirim ke Receiver.
  4. Receiver (berjalan sebagai akun Google Anda) menyimpan ke folder Drive.

Konfigurasi lewat GitHub Secrets / Variables:
  PTREE_USER, PTREE_PASSWORD, RECEIVER_URL, RECEIVER_TOKEN,
  REMOTE_TEMPLATE, LOOKBACK_HOURS, MAX_UPLOADS_PER_RUN, FTP_HOST
"""

import ftplib
import io
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import sys

FILE_PATTERN = re.compile(r"^H09_(\d{8})_(\d{4})_.*L2WLF.*\.csv$", re.IGNORECASE)


def cfg():
    c = {
        "host": (os.environ.get("FTP_HOST") or "ftp.ptree.jaxa.jp"),
        "port": int(os.environ.get("FTP_PORT") or "21"),
        "user": os.environ.get("PTREE_USER", ""),
        "password": os.environ.get("PTREE_PASSWORD", "").strip(),
        "receiver_url": os.environ.get("RECEIVER_URL", ""),
        "token": os.environ.get("RECEIVER_TOKEN", "").strip(),
        "template": (os.environ.get("REMOTE_TEMPLATE") or
                     "/pub/himawari/L2/WLF/010/{yyyymm}/{dd}/{hh}/").strip(),
        "lookback": int(os.environ.get("LOOKBACK_HOURS") or "6"),
        "max_uploads": int(os.environ.get("MAX_UPLOADS_PER_RUN") or "60"),
    }
    missing = [k for k in ("user", "password", "receiver_url", "token") if not c[k]]
    if missing:
        raise RuntimeError("Konfigurasi belum lengkap: " + ", ".join(missing))
    return c


# ----------------------------------------------------------------------
# RECEIVER (Apps Script web app)
# ----------------------------------------------------------------------
def receiver(c, payload):
    payload = dict(payload)
    payload["token"] = c["token"]
    req = urllib.request.Request(
        c["receiver_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode("utf-8", "replace")
            try:
                data = json.loads(body)
            except ValueError:
                raise RuntimeError(
                    "Receiver tidak mengembalikan JSON. Pastikan web app Receiver "
                    "di-deploy dengan akses 'Siapa saja' (Anyone). Awal respons: " + body[:200])
            if not data.get("ok"):
                raise RuntimeError("Receiver menolak: " + str(data.get("error")))
            return data
        except Exception as e:  # noqa
            last = e
            if "menolak" in str(e) or "JSON" in str(e):
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("Receiver tidak dapat dihubungi: %s" % last)


# ----------------------------------------------------------------------
# FTP
# ----------------------------------------------------------------------
def connect(c):
    last = None
    for attempt in range(1, 4):
        try:
            ftp = ftplib.FTP(timeout=60)
            ftp.connect(c["host"], c["port"])
            ftp.login(c["user"], c["password"])
            ftp.set_pasv(True)
            return ftp
        except Exception as e:  # noqa
            last = e
            time.sleep(4 * attempt)
    raise RuntimeError("Login FTP JAXA gagal: %s" % last)


def list_dir(ftp, path):
    try:
        return [n.rsplit("/", 1)[-1] for n in ftp.nlst(path)]
    except ftplib.error_perm as e:
        if "550" in str(e):
            return []
        raise


def remote_dirs(c, now_utc):
    out = []
    for h in range(c["lookback"], -1, -1):
        t = now_utc - timedelta(hours=h)
        out.append(c["template"].format(
            yyyy=t.strftime("%Y"), mm=t.strftime("%m"), yyyymm=t.strftime("%Y%m"),
            dd=t.strftime("%d"), hh=t.strftime("%H"), yyyymmdd=t.strftime("%Y%m%d")))
    return out


# ----------------------------------------------------------------------
# JOB
# ----------------------------------------------------------------------
def job():
    started = time.time()
    c = cfg()
    now_utc = datetime.now(timezone.utc)

    existing = set(receiver(c, {"action": "list"}).get("files", []))
    print("Receiver: %d file sudah ada di Drive." % len(existing))

    ftp = connect(c)
    uploaded, failed, seen = [], [], 0
    try:
        for d in remote_dirs(c, now_utc):
            for name in sorted(n for n in list_dir(ftp, d) if FILE_PATTERN.match(n)):
                seen += 1
                if name in existing:
                    continue
                if len(uploaded) >= c["max_uploads"] or time.time() - started > 240:
                    break
                try:
                    buf = io.BytesIO()
                    ftp.retrbinary("RETR " + d.rstrip("/") + "/" + name, buf.write)
                    content = buf.getvalue().decode("utf-8", "replace")
                    if not content.strip():
                        raise RuntimeError("file kosong")
                    receiver(c, {"action": "upload", "name": name, "content": content})
                    uploaded.append(name)
                    print("Diunggah: " + name)
                except Exception as e:  # noqa
                    failed.append(name)
                    print("GAGAL %s: %s" % (name, e))
    finally:
        try:
            ftp.quit()
        except Exception:  # noqa
            pass

    latest = sorted(existing.union(uploaded))
    latest_name = latest[-1] if latest else None
    age_min = None
    if latest_name:
        m = FILE_PATTERN.match(latest_name)
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        age_min = round((now_utc - t).total_seconds() / 60)

    result = {
        "ok": not failed,
        "files_on_ftp_window": seen,
        "uploaded": len(uploaded),
        "failed": failed,
        "latest_file": latest_name,
        "latest_age_minutes": age_min,
        "seconds": round(time.time() - started, 1),
    }
    if seen == 0:
        result["warning"] = ("Tidak ada file WLF di FTP untuk %d jam terakhir. "
                             "Periksa REMOTE_TEMPLATE." % c["lookback"])
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    try:
        res = job()
    except Exception as e:  # noqa
        print("ERROR: %s" % e)
        sys.exit(1)
    if res.get("warning"):
        print("PERINGATAN: " + res["warning"])
    # Exit code 1 -> GitHub menandai run gagal dan mengirim email notifikasi
    sys.exit(0 if res.get("ok") else 1)
