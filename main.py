import json, os, shutil, subprocess, threading, uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "base")
FORMATS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}
JOBS = {}
_whisper = None
app = FastAPI()


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-500:])
    return p.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)]))


def transcribe(path):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_SIZE, compute_type="int8")
    segs, _ = _whisper.transcribe(str(path), word_timestamps=True, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip(),
             "words": [(w.start, w.end, w.word.strip()) for w in (s.words or [])]} for s in segs]


def pick_moments(segs, total, n, max_len):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    tr = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs)[:120000]
    prompt = (f"You are a viral short-form video editor. Transcript with timestamps in seconds:\n{tr}\n\n"
              f"Pick the {n} best moments for TikTok/Reels/Shorts. Rules: each 20-{max_len} seconds, "
              f"inside 0-{total:.0f}s, start and end on sentence boundaries, strong hook in the first 3 seconds, "
              "self-contained, no overlaps. Return ONLY a JSON array of objects with keys: "
              "start (number), end (number), score (0-100 viral potential), title (short), "
              "hook (opening caption line), reason (one sentence). Write title/hook/reason in the transcript's language.")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_KEY},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}},
        timeout=180)
    r.raise_for_status()
    items = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    clean = []
    for it in items:
        try:
            a, b = max(0.0, float(it["start"])), min(total, float(it["end"]))
            if b - a < 10:
                continue
            b = min(b, a + max_len)
            clean.append({"start": a, "end": b, "score": int(it.get("score", 0)),
                          "title": str(it.get("title", "Clip")), "hook": str(it.get("hook", "")),
                          "reason": str(it.get("reason", ""))})
        except (KeyError, ValueError, TypeError):
            continue
    clean.sort(key=lambda c: -c["score"])
    picked = []
    for c in clean:  # drop overlaps, keep best-scoring
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return sorted(picked, key=lambda c: c["start"])


def _ts(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def make_ass(segs, a, b, w, h, path):
    words = [x for s in segs for x in s["words"] if x[0] >= a - 0.05 and x[1] <= b + 0.05]
    fs = int(min(w, h) * 0.075)
    mv = int(h * 0.2) if h > w else int(h * 0.07)
    head = ("[Script Info]\nScriptType: v4.00+\n" f"PlayResX: {w}\nPlayResY: {h}\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
            "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
            f"MarginV,Encoding\nStyle: Default,Arial,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,"
            f"100,100,0,0,1,{max(3, fs // 14)},2,2,60,60,{mv},1\n\n[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines = []
    for i in range(0, len(words), 3):
        grp = words[i:i + 3]
        start = grp[0][0] - a
        nxt = words[i + 3][0] - a if i + 3 < len(words) else grp[-1][1] - a
        end = max(start + 0.2, min(nxt, grp[-1][1] - a + 0.4))
        txt = " ".join(x[2] for x in grp).upper()
        lines.append(f"Dialogue: 0,{_ts(max(0, start))},{_ts(end)},Default,,0,0,0,,{txt}")
    path.write_text(head + "\n".join(lines), encoding="utf-8")


def render(src, a, b, w, h, ass, outdir, name):
    r = f"{w / h:.4f}"
    vf = (f"crop=w='if(gt(iw/ih,{r}),ih*{r},iw)':h='if(gt(iw/ih,{r}),ih,iw/{r})',scale={w}:{h},setsar=1")
    if ass:
        vf += f",ass={ass.name}"
    run(["ffmpeg", "-y", "-ss", f"{a:.2f}", "-i", str(src), "-t", f"{b - a:.2f}", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", name], cwd=outdir)


def pipeline(jid, url, fmt, n, max_len, captions):
    d = DATA / jid
    job = JOBS[jid]
    try:
        w, h = FORMATS[fmt]
        if url:
            job.update(stage="Downloading video", progress=5)
            run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4", "--no-playlist",
                 "-o", str(d / "source.%(ext)s"), url])
        src = next(d.glob("source.*"))
        total = duration(src)
        n = n or max(1, min(15, round(total / 60)))
        job.update(stage="Transcribing audio", progress=20)
        segs = transcribe(src)
        if not segs:
            raise RuntimeError("No speech found in this video")
        job.update(stage="Finding the best moments", progress=45)
        moments = pick_moments(segs, total, n, max_len)
        clips = []
        for i, m in enumerate(moments, 1):
            job.update(stage=f"Rendering clip {i} of {len(moments)}", progress=50 + int(48 * (i - 1) / len(moments)))
            ass = None
            if captions:
                ass = d / f"clip_{i:02d}.ass"
                make_ass(segs, m["start"], m["end"], w, h, ass)
            name = f"clip_{i:02d}.mp4"
            render(src, m["start"], m["end"], w, h, ass, d, name)
            clips.append({**m, "file": f"/files/{jid}/{name}"})
        job.update(stage="Done", progress=100, status="done", clips=clips)
    except Exception as e:
        job.update(status="error", stage="Failed", error=str(e))


@app.post("/api/jobs")
async def create_job(url: str = Form(""), fmt: str = Form("9:16"), clips: int = Form(0),
                     max_len: int = Form(60), captions: bool = Form(True),
                     file: Optional[UploadFile] = File(None)):
    if fmt not in FORMATS:
        raise HTTPException(400, "Unknown format")
    if not url and not file:
        raise HTTPException(400, "Add a video link or upload a file")
    jid = uuid.uuid4().hex[:10]
    (DATA / jid).mkdir()
    if file:
        with open(DATA / jid / "source.mp4", "wb") as f:
            shutil.copyfileobj(file.file, f)
    JOBS[jid] = {"id": jid, "status": "running", "stage": "Starting", "progress": 1, "clips": []}
    threading.Thread(target=pipeline, args=(jid, url.strip(), fmt, clips, max(20, min(90, max_len)), captions),
                     daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[jid]


app.mount("/files", StaticFiles(directory=DATA), name="files")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
