# src/ingest.py
from yt_dlp import YoutubeDL
import hashlib, shutil, sqlite3, subprocess, sys
from pathlib import Path
import cv2, imagehash, yaml
import yt_dlp

CFG = yaml.safe_load(Path("config.yaml").read_text())
P = CFG["paths"]
DB = Path(P["db"])
RAW = Path(P["raw_root"])
RAW.mkdir(parents=True, exist_ok=True)


def db():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS videos(
        url TEXT PRIMARY KEY, hash TEXT, path TEXT,
        duration REAL, fps REAL, w INT, h INT, size_mb REAL,
        status TEXT, fail_reason TEXT, created_at TEXT DEFAULT (datetime('now'))
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS dedup(
        hash TEXT PRIMARY KEY, keeper_url TEXT
    )""")
    return c


def already(url):
    return db().execute("SELECT 1 FROM videos WHERE url=?", (url,)).fetchone()


def fetch(url):
    """yt-dlp 下载，返回本地路径"""
    opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": str(RAW / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "retries": 3,
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
        filename = y.prepare_filename(info)
        p = Path(filename)
        if not p.exists() and p.with_suffix(".mp4").exists():
            filename = str(p.with_suffix(".mp4"))
        return filename, info


def fetch_piexls(url):
    opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(RAW / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "retries": 3,
    }
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
        return y.prepare_filename(info), info

def normalize(src):
    """ffmpeg 统一制式；AV1 等无硬解格式会被软件解码，慢但稳"""
    dst = src.with_name(src.stem + "_norm.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={CFG['video']['target_w']}:{CFG['video']['target_h']}:force_original_aspect_ratio=decrease,"
               f"pad={CFG['video']['target_w']}:{CFG['video']['target_h']}:(ow-iw)/2:(oh-ih)/2,fps={CFG['video']['target_fps']}",
        "-c:v", "h264_videotoolbox" if sys.platform == "darwin" else "libx264",
        "-preset", "fast", "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def probe(path):
    cap = cv2.VideoCapture(str(path))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dur, fps, w, h


def phash(path):
    # 抽 8 帧做平均感知哈希，比单帧稳
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(n * i / 8) for i in range(8)]
    hs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            hs.append(imagehash.average_hash(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), hash_size=16))
    cap.release()
    return sum(hs, None) / len(hs) if hs else None


def ingest_one(url):
    conn = db()
    if already(url):
        print(f"SKIP {url}")
        return
    try:
        local, info = fetch(url)
        norm = normalize(Path(local))
        dur, fps, w, h = probe(norm)
        if not (CFG["video"]["min_duration_s"] <= dur <= CFG["video"]["max_duration_s"]):
            raise ValueError(f"duration {dur}s out of range")
        size_mb = Path(norm).stat().st_size / 1048576
        if size_mb > CFG["video"]["max_file_mb"]:
            raise ValueError(f"size {size_mb:.1f}MB too big")
        hsh = str(phash(norm))
        # 去重
        row = conn.execute("SELECT keeper_url FROM dedup WHERE hash=?", (hsh,)).fetchone()
        if row:
            print(f"DEDUP {url} ~= {row[0]}")
            conn.execute("INSERT OR REPLACE INTO videos VALUES(?,?,?,?,?,?,?,?,'dup','')",
                         (url, hsh, str(norm), dur, fps, w, h, round(size_mb, 2)))
        else:
            conn.execute("INSERT INTO dedup VALUES(?,?)", (hsh, url))
            conn.execute("INSERT OR REPLACE INTO videos VALUES(?,?,?,?,?,?,?,?,'ok','')",
                         (url, hsh, str(norm), dur, fps, w, h, round(size_mb, 2)))
        conn.commit()
        print(f"OK {url} -> {norm.name}  {dur:.1f}s {w}x{h}")
    except Exception as e:
        conn.execute("INSERT OR REPLACE INTO videos(url,status,fail_reason) VALUES(?,?,?)",
                     (url, "fail", str(e)[:200]))
        conn.commit()
        print(f"FAIL {url}: {e}")


def ingest_list(txt):
    urls = [l.strip() for l in Path(txt).read_text().splitlines() if l.strip() and not l.startswith("#")]
    for u in urls:
        ingest_one(u)
    stat = db().execute("SELECT status,COUNT(*) FROM videos GROUP BY 1").fetchall()
    print("统计:", dict(stat))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="assets/sources.txt")
    ap.add_argument("--url")
    args = ap.parse_args()
    if args.url:
        ingest_one(args.url)
    else:
        ingest_list(args.list)