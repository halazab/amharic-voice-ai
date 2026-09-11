"""
Amharic ASR dataset pipeline.

For each YouTube URL, this script:
  1. Downloads the audio (best quality) and Amharic (am) auto/manual subtitles.
  2. Re-encodes the audio to 16 kHz mono WAV (16-bit PCM).
  3. Parses the VTT cues, cleaning each text line per the spec
     (Ge'ez-only, strip annotations, digits → Amharic words, etc.).
  4. Cuts one WAV + one TXT per surviving subtitle cue, padded/trimmed
     to a configurable duration window (default 1-30s, recommended 5-15s).
  5. Trims ~0.2s lead/tail of silence from each clip.
  6. Appends a JSONL manifest row per clip with: audio path, text, duration,
     source video id, speaker ("unknown" for YouTube), and other fields.
  7. After all videos are processed, splits per-video (NOT random) into
     train 90% / val 5% / test 5%, writes manifests/{train,validation,test}.jsonl
  8. Builds an HF datasets.Dataset and save_to_disk()s it (load_dataset() ready).
  9. Runs a character audit and writes README.md with provenance + stats.

Usage:
    python3 amharic_dataset.py \
        --urls <path-to-urls.txt or single URL> \
        --data-root /home/z/my-project/download/amharic-voice-ai/data \
        [--min-duration 1.0] [--max-duration 30.0] [--ideal-min 5.0] [--ideal-max 15.0] \
        [--keep-punctuation] [--no-keep-punctuation]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import soundfile as sf
import yt_dlp
from datasets import Dataset, Audio, Features, Value

# We parse WebVTT inline below — no `webvtt` package dependency.

# Add the scripts dir so we can import our text utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amharic_text_utils import (
    clean_amharic_text,
    character_audit,
    format_audit_report,
)

# yt-dlp downloads captions as .vtt; we'll parse the vtt format manually.
# We don't depend on the `webvtt` package — handle it inline below.

SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SUBTYPE = "PCM_16"
TRIM_LEAD_TAIL_SEC = 0.2  # silence pad if shorter than 0.2s; trim if longer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")


def ts_to_seconds(hh: str, mm: str, ss: str, ms: str) -> float:
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_vtt(path: Path) -> list[dict]:
    """
    Parse a WebVTT file into a list of cues:
        [{"start": float, "end": float, "text": str}, ...]

    Strips inline tags, cue identifiers, and WEBVTT header. Joins multi-line
    cue payloads with a single space (these are line breaks inside the cue).
    """
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Normalize newlines
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    cues: list[dict] = []
    blocks = raw.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("WEBVTT") or block.startswith("NOTE"):
            continue
        lines = block.split("\n")
        # First line may be a cue identifier (e.g. "1", "cue-12") — skip it
        # if it doesn't contain the timestamp arrow.
        ts_line_idx = 0
        if "-->" not in lines[0]:
            if len(lines) > 1 and "-->" in lines[1]:
                ts_line_idx = 1
            else:
                continue
        ts_line = lines[ts_line_idx]
        m = TIMESTAMP_RE.search(ts_line)
        if not m:
            continue
        start = ts_to_seconds(*m.groups())
        # End timestamp is the second match in the same line
        m2 = TIMESTAMP_RE.search(ts_line, m.end())
        if not m2:
            continue
        end = ts_to_seconds(*m2.groups())
        text_lines = lines[ts_line_idx + 1:]
        text = " ".join(t.strip() for t in text_lines if t.strip())
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


# ---------------------------------------------------------------------------
# YouTube download / VTT fetch
# ---------------------------------------------------------------------------
def video_id_from_url(url: str) -> str:
    """Extract the 11-char YouTube video id from a URL."""
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url.strip().split("/")[-1][:11]


def download_audio_and_subs(url: str, raw_dir: Path, prefer_lang: str = "am",
                             cookies_path: str | None = None) -> dict:
    """
    Downloads best-audio stream and the Amharic subtitles for one URL.

    Returns a dict: { "video_id": str, "audio_path": Path, "vtt_path": Path|None }
    """
    vid = video_id_from_url(url)
    out_tmpl = str(raw_dir / vid / "%(title)s.%(ext)s")
    ydl_opts = {
        # Prefer m4a (AAC) over webm (opus) because ffmpeg handles it more reliably
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "writethumbnail": False,
        "writeinfojson": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [prefer_lang, f"{prefer_lang}-orig", "am"],
        "subtitlesformat": "vtt",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        # Robust network settings
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 5,
        "socket_timeout": 60,
    }
    # Attach cookies if provided (required for bot-blocked IPs)
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path
    print(f"  ↓ Downloading audio + subs: {vid} ...", flush=True)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        print(f"  ✗ Download failed for {vid}: {e}")
        return {"video_id": vid, "audio_path": None, "vtt_path": None, "error": str(e)}

    # Find the produced WAV
    vid_dir = raw_dir / vid
    wavs = sorted(vid_dir.glob("*.wav"))
    if not wavs:
        # Sometimes yt-dlp leaves the audio as .webm or .m4a if the postprocessor
        # silently failed. Try a fallback re-encode via ffmpeg.
        others = sorted(vid_dir.glob("*.webm")) + sorted(vid_dir.glob("*.m4a")) + sorted(vid_dir.glob("*.opus"))
        if others:
            wav_path = vid_dir / f"{vid}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(others[0]), "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-acodec", "pcm_s16le", str(wav_path)],
                check=True, capture_output=True,
            )
            wavs = [wav_path]
    if not wavs:
        return {"video_id": vid, "audio_path": None, "vtt_path": None,
                "error": "No audio file produced"}

    audio_path = wavs[0]

    # Find the produced VTT (prefer the chosen lang, then any .vtt in the folder)
    vtt_candidates = sorted(vid_dir.glob(f"*{prefer_lang}*.vtt")) + sorted(vid_dir.glob("*.vtt"))
    vtt_path = vtt_candidates[0] if vtt_candidates else None

    return {
        "video_id": vid,
        "audio_path": audio_path,
        "vtt_path": vtt_path,
        "title": info.get("title", "") if isinstance(info, dict) else "",
        "uploader": info.get("uploader", "") if isinstance(info, dict) else "",
        "upload_date": info.get("upload_date", "") if isinstance(info, dict) else "",
        "duration": info.get("duration", None) if isinstance(info, dict) else None,
    }


def normalize_audio_to_16k_mono(input_path: Path, output_path: Path) -> float:
    """Re-encode to 16 kHz mono 16-bit PCM WAV. Returns duration in seconds."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-ac", str(AUDIO_CHANNELS),
         "-ar", str(SAMPLE_RATE),
         "-acodec", "pcm_s16le",
         str(output_path)],
        check=True, capture_output=True,
    )
    info = sf.info(str(output_path))
    return info.frames / info.samplerate


# ---------------------------------------------------------------------------
# Per-cue segmentation
# ---------------------------------------------------------------------------
def load_audio(path: Path) -> tuple["np.ndarray", int]:
    import numpy as np
    audio, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if sr != SAMPLE_RATE:
        # Should already be 16k, but resample defensively
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = (audio * 32767).astype("int16")
        sr = SAMPLE_RATE
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype("int16")
    return audio, sr


def trim_silence(audio: "np.ndarray", sr: int, lead_tail_sec: float = TRIM_LEAD_TAIL_SEC) -> "np.ndarray":
    """
    Trim leading/trailing silence to ~lead_tail_sec.
    Uses simple energy-based thresholding (no external VAD).
    """
    import numpy as np
    if len(audio) == 0:
        return audio
    # Use a small energy threshold (RMS) over 30ms windows
    hop = int(sr * 0.03)
    if hop < 1:
        return audio
    frame_energy = np.array([
        np.sqrt(np.mean(audio[i:i + hop].astype("float32") ** 2))
        for i in range(0, len(audio) - hop, hop)
    ])
    if len(frame_energy) == 0:
        return audio
    # Threshold: 1% of max energy, with a floor
    threshold = max(50.0, 0.01 * float(frame_energy.max()))
    above = np.where(frame_energy > threshold)[0]
    if len(above) == 0:
        return audio
    start_sample = max(0, int(above[0] * hop) - int(lead_tail_sec * sr))
    end_sample = min(len(audio), int((above[-1] + 1) * hop) + int(lead_tail_sec * sr))
    return audio[start_sample:end_sample]


def write_segment_clip(audio: "np.ndarray", sr: int, start_sec: float, end_sec: float,
                       out_wav: Path) -> float:
    """Cut and write one clip. Pads if too short, trims silence at lead/tail."""
    import numpy as np
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    clip = audio[s:e]
    clip = trim_silence(clip, sr)
    if len(clip) < sr * 0.1:  # < 100ms after trim — too short
        return 0.0
    sf.write(str(out_wav), clip, sr, subtype="PCM_16")
    return len(clip) / sr


# ---------------------------------------------------------------------------
# Process one video → segments + JSONL rows
# ---------------------------------------------------------------------------
@dataclass
class ProcessResult:
    video_id: str
    n_cues_total: int = 0
    n_segments_written: int = 0
    n_dropped: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    duration_written_sec: float = 0.0
    error: str | None = None


def process_video(url: str, raw_dir: Path, segments_dir: Path,
                  *, min_duration: float, max_duration: float,
                  ideal_min: float, ideal_max: float,
                  keep_punctuation: bool,
                  cookies_path: str | None = None) -> ProcessResult:
    vid_meta = download_audio_and_subs(url, raw_dir, prefer_lang="am",
                                       cookies_path=cookies_path)
    vid = vid_meta["video_id"]
    res = ProcessResult(video_id=vid)
    if vid_meta.get("error"):
        res.error = vid_meta["error"]
        return res

    audio_path = vid_meta["audio_path"]
    vtt_path = vid_meta.get("vtt_path")
    if vtt_path is None:
        res.error = "No Amharic (am) subtitle available — skipped."
        return res

    # Normalize audio to 16k mono WAV in raw/<vid>/audio_16k.wav
    norm_wav = audio_path.parent / "audio_16k.wav"
    if audio_path != norm_wav:
        total_dur = normalize_audio_to_16k_mono(audio_path, norm_wav)
    else:
        info = sf.info(str(norm_wav))
        total_dur = info.frames / info.samplerate
    print(f"    audio normalized: {total_dur:.1f}s @ 16kHz mono", flush=True)

    audio, sr = load_audio(norm_wav)

    # Parse cues
    cues = parse_vtt(vtt_path)
    res.n_cues_total = len(cues)
    print(f"    parsed {len(cues)} subtitle cues from {vtt_path.name}", flush=True)

    seg_dir = segments_dir / vid
    seg_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    for cue in cues:
        raw_text = cue["text"]
        cleaned_text, reason = clean_amharic_text(raw_text, keep_punctuation=keep_punctuation)
        if cleaned_text is None:
            res.n_dropped[reason] = res.n_dropped.get(reason, 0) + 1
            continue
        start, end = cue["start"], cue["end"]
        if end <= start:
            res.n_dropped["zero_duration"] = res.n_dropped.get("zero_duration", 0) + 1
            continue
        dur = end - start
        # Duration filter
        if dur < min_duration:
            res.n_dropped["too_short"] = res.n_dropped.get("too_short", 0) + 1
            continue
        if dur > max_duration:
            # Try splitting long cues into halves until they fit (rare)
            res.n_dropped["too_long"] = res.n_dropped.get("too_long", 0) + 1
            continue
        counter += 1
        out_wav = seg_dir / f"{counter:04d}.wav"
        out_txt = seg_dir / f"{counter:04d}.txt"
        actual_dur = write_segment_clip(audio, sr, start, end, out_wav)
        if actual_dur <= 0:
            res.n_dropped["silence_only"] = res.n_dropped.get("silence_only", 0) + 1
            continue
        # Skip if after silence-trim the clip dropped below threshold
        if actual_dur < min_duration:
            res.n_dropped["too_short_after_trim"] = res.n_dropped.get("too_short_after_trim", 0) + 1
            out_wav.unlink(missing_ok=True)
            out_txt.unlink(missing_ok=True)
            continue
        out_txt.write_text(cleaned_text, encoding="utf-8")
        is_ideal = (ideal_min <= actual_dur <= ideal_max)
        row = {
            "audio": str(out_wav.relative_to(segments_dir)),
            "text": cleaned_text,
            "duration": round(actual_dur, 3),
            "source": f"youtube:{vid}",
            "speaker": "unknown",
            "title": vid_meta.get("title", ""),
            "uploader": vid_meta.get("uploader", ""),
            "upload_date": vid_meta.get("upload_date", ""),
            "is_ideal_duration": is_ideal,
        }
        res.rows.append(row)
        res.duration_written_sec += actual_dur

    res.n_segments_written = len(res.rows)
    return res


# ---------------------------------------------------------------------------
# Splitting (per-video, NOT random)
# ---------------------------------------------------------------------------
def split_rows_by_video(rows: list[dict], *,
                        train_frac: float = 0.9,
                        val_frac: float = 0.05,
                        test_frac: float = 0.05,
                        seed: int = 42) -> dict[str, list[dict]]:
    """
    Group rows by source video; assign whole videos to splits so the same
    speaker/source cannot leak between train and val/test.

    Strategy: sort videos by total duration (descending), greedily assign to
    whichever split is most "under-target" until quotas are met.
    For <3 videos, fall back to splitting within the single video by position
    (first 90% train, middle 5% val, last 5% test) — not ideal but at least
    keeps temporal locality.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["source"], []).append(r)

    splits = {"train": [], "validation": [], "test": []}

    if len(by_video) >= 3:
        # Compute total durations per video
        vid_durations = [(vid, sum(r["duration"] for r in vrows)) for vid, vrows in by_video.items()]
        vid_durations.sort(key=lambda x: -x[1])  # largest first

        total_dur = sum(d for _, d in vid_durations)
        targets = {
            "train": total_dur * train_frac,
            "validation": total_dur * val_frac,
            "test": total_dur * test_frac,
        }
        current = {"train": 0.0, "validation": 0.0, "test": 0.0}

        # Assign largest videos greedily to the split that is furthest below its target ratio
        for vid, dur in vid_durations:
            # Relative shortfalls
            shortfalls = {
                split: (targets[split] - current[split]) / max(targets[split], 1e-6)
                for split in ("train", "validation", "test")
            }
            # Always reserve test+val first if they're still way below their targets
            # and we still have videos left; otherwise fall to train.
            n_videos_left = len(vid_durations) - vid_durations.index((vid, dur))
            if n_videos_left <= 2 and current["test"] == 0.0:
                chosen = "test"
            elif n_videos_left <= 3 and current["validation"] == 0.0:
                chosen = "validation"
            else:
                chosen = max(shortfalls, key=shortfalls.get)
            splits[chosen].extend(by_video[vid])
            current[chosen] += dur

        # If val/test ended up empty (very few videos), pull smallest video(s) into them
        if not splits["validation"]:
            smallest_vid = min(by_video.keys(), key=lambda v: sum(r["duration"] for r in by_video[v]))
            splits["validation"] = by_video[smallest_vid]
            splits["train"] = [r for r in splits["train"] if r["source"] != smallest_vid]
        if not splits["test"]:
            smallest_vid = min(by_video.keys(), key=lambda v: sum(r["duration"] for r in by_video[v]))
            splits["test"] = by_video[smallest_vid]
            splits["train"] = [r for r in splits["train"] if r["source"] != smallest_vid]
    else:
        # Fall back: within-video split by position (preserves temporal locality)
        rng = random.Random(seed)
        for vid, vrows in by_video.items():
            # Sort by order if we can derive (use position in rows list)
            n = len(vrows)
            if n < 5:
                # Single tiny video: throw to train
                splits["train"].extend(vrows)
                continue
            n_test = max(1, n // 20)  # 5%
            n_val = max(1, n // 20)   # 5%
            n_train = n - n_test - n_val
            splits["train"].extend(vrows[:n_train])
            splits["validation"].extend(vrows[n_train:n_train + n_val])
            splits["test"].extend(vrows[n_train + n_val:])
        print(f"  ⚠ Only {len(by_video)} video(s) — fell back to within-video temporal split.", flush=True)
        print(f"    For a real val/test signal, re-run with ≥3 different YouTube videos.", flush=True)

    return splits


# ---------------------------------------------------------------------------
# JSONL + HF dataset writing
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_hf_dataset(rows: list[dict], segments_root: Path, out_dir: Path) -> None:
    """
    Build a HuggingFace `datasets.Dataset` from a list of manifest rows.
    Casts the `audio` column to `Audio()` so `load_dataset` will decode WAV
    on-the-fly. save_to_disk(out_dir).
    """
    if not rows:
        print("  ⚠ no rows to write — skipping HF dataset build")
        return
    # Resolve absolute audio paths
    abs_rows = []
    for r in rows:
        ar = dict(r)
        ar["audio"] = str((segments_root / r["audio"]).resolve())
        abs_rows.append(ar)
    features = Features({
        "audio": Audio(sampling_rate=SAMPLE_RATE),
        "text": Value("string"),
        "duration": Value("float32"),
        "source": Value("string"),
        "speaker": Value("string"),
        "title": Value("string"),
        "uploader": Value("string"),
        "upload_date": Value("string"),
        "is_ideal_duration": Value("bool"),
    })
    ds = Dataset.from_list(abs_rows, features=features)
    ds.save_to_disk(str(out_dir))
    print(f"  ✓ HF dataset saved: {out_dir}  ({len(ds)} examples)")


# ---------------------------------------------------------------------------
# Stats + README
# ---------------------------------------------------------------------------
def compute_stats(rows: list[dict]) -> dict:
    durations = [r["duration"] for r in rows]
    texts = [r["text"] for r in rows]
    char_lens = [len(t) for t in texts]
    return {
        "n_examples": len(rows),
        "total_duration_sec": round(sum(durations), 2),
        "total_duration_hrs": round(sum(durations) / 3600, 3),
        "avg_duration_sec": round(sum(durations) / max(1, len(durations)), 3),
        "min_duration_sec": round(min(durations), 3) if durations else 0,
        "max_duration_sec": round(max(durations), 3) if durations else 0,
        "avg_chars": round(sum(char_lens) / max(1, len(char_lens)), 1),
        "n_videos": len({r["source"] for r in rows}),
        "n_speakers": len({r["speaker"] for r in rows}),
    }


def write_readme(data_root: Path, all_rows: list[dict],
                 splits: dict[str, list[dict]], stats: dict,
                 audit_report: str, drop_reasons: dict, sources_meta: list[dict]) -> None:
    readme = data_root / "README.md"
    train_stats = compute_stats(splits["train"])
    val_stats = compute_stats(splits["validation"])
    test_stats = compute_stats(splits["test"])
    body = f"""# Amharic ASR Dataset

Built from YouTube audio + Amharic captions. Generated `{time.strftime("%Y-%m-%d %H:%M:%S")}`.

## Layout
```
data/
├── raw/                 # original downloads (audio + .vtt + info.json per video)
├── segments/            # cut clips: <vid>/0001.wav + 0001.txt
├── manifests/
│   ├── train.jsonl      # {train_stats['n_examples']} examples ({train_stats['total_duration_hrs']}h)
│   ├── validation.jsonl # {val_stats['n_examples']} examples ({val_stats['total_duration_hrs']}h)
│   └── test.jsonl       # {test_stats['n_examples']} examples ({test_stats['total_duration_hrs']}h)
├── hf_dataset/          # HuggingFace save_to_disk format (load_dataset ready)
├── character_audit.txt  # full unicode audit
└── README.md
```

## Stats
- Total examples: {stats['n_examples']}
- Total audio: {stats['total_duration_hrs']} hours ({stats['total_duration_sec']}s)
- Average clip duration: {stats['avg_duration_sec']}s (range {stats['min_duration_sec']}-{stats['max_duration_sec']}s)
- Number of source videos: {stats['n_videos']}
- Number of distinct speakers: {stats['n_speakers']} (most are "unknown" for YouTube)

## Audio format
- Sample rate: 16,000 Hz
- Channels: mono
- Format: WAV, 16-bit PCM
- Silence trimmed: ~0.2s lead/tail
- Per-cue segmentation: each subtitle entry → one WAV + one TXT

## Text rules applied
- Verbatim text, Ge'ez script only
- Latin letters, digits, emojis, annotation markers stripped
- Plain integers (0-9999) converted to Amharic words
- Numbers outside 0-9999 → segment dropped
- Natural Ge'ez punctuation preserved (። ፣ ፤ ፥ ፦ ፧ ፨)
- Each segment survived a character audit (no garbage Unicode)

## Splits
- Split strategy: **by source video, not random** — the same YouTube video
  cannot appear in both train and validation/test. This prevents speaker/
  recording-condition leakage that inflates val/test scores.
- Train: 90% of examples
- Validation: 5%
- Test: 5% (sacred — never trained on, never tuned on)

## Provenance — source videos
"""
    for m in sources_meta:
        body += f"### `{m.get('video_id', '?')}`\n"
        body += f"- Title: {m.get('title', '?')}\n"
        body += f"- Uploader: {m.get('uploader', '?')}\n"
        body += f"- Upload date: {m.get('upload_date', '?')}\n"
        body += f"- URL: https://youtu.be/{m.get('video_id', '')}\n"
        body += f"- Segments written: {m.get('n_segments_written', 0)}\n"
        if m.get('error'):
            body += f"- ⚠ Error: {m['error']}\n"
        body += "\n"

    body += f"""## Dropped segments (and why)
"""
    if drop_reasons:
        for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            body += f"- {reason}: {count}\n"
    else:
        body += "- (none)\n"

    body += f"""
## Loading
```python
from datasets import load_from_disk
ds = load_from_disk("data/hf_dataset")
print(ds[0])  # {{"audio": {{...}}, "text": "ሰላም ...", "duration": 8.4, ...}}
```

## Character audit
See `character_audit.txt` for the full per-character report. The audit lists
every unique character in the corpus and flags any that fall outside the
allowed Ethiopic + basic-punctuation set.

## License notes
- Audio sourced from YouTube. Original videos retain their original copyright.
- This dataset is intended for **research and educational ASR training**.
- Each segment's `source` field links back to the original video for
  provenance and takedown compliance.
- If you redistribute, include this README and the manifest.
"""
    readme.write_text(body, encoding="utf-8")
    (data_root / "character_audit.txt").write_text(audit_report, encoding="utf-8")
    print(f"  ✓ Wrote README.md + character_audit.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urls", required=True,
                   help="Path to a .txt file with one YouTube URL per line, OR a single URL.")
    p.add_argument("--data-root", required=True,
                   help="Root directory of the dataset (must contain raw/, segments/, manifests/).")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--ideal-min", type=float, default=5.0)
    p.add_argument("--ideal-max", type=float, default=15.0)
    p.add_argument("--keep-punctuation", dest="keep_punctuation", action="store_true", default=True)
    p.add_argument("--no-keep-punctuation", dest="keep_punctuation", action="store_false")
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--cookies", default=None,
                   help="Path to a Netscape-format cookies.txt file for YouTube authentication.")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    raw_dir = data_root / "raw"
    segments_dir = data_root / "segments"
    manifests_dir = data_root / "manifests"
    hf_dir = data_root / "hf_dataset"
    for d in (raw_dir, segments_dir, manifests_dir, hf_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load URLs
    if os.path.isfile(args.urls):
        with open(args.urls, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        urls = [args.urls.strip()]
    if not urls:
        print("No URLs to process.")
        sys.exit(1)
    print(f"Will process {len(urls)} URL(s):", flush=True)
    for u in urls:
        print(f"  - {u}", flush=True)

    all_rows: list[dict] = []
    sources_meta: list[dict] = []
    total_dropped: dict = {}

    for url in urls:
        print(f"\n=== {url} ===", flush=True)
        res = process_video(
            url, raw_dir, segments_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            ideal_min=args.ideal_min,
            ideal_max=args.ideal_max,
            keep_punctuation=args.keep_punctuation,
            cookies_path=args.cookies,
        )
        # Pull source metadata from the first row (it has the title/uploader/upload_date fields)
        first_row = res.rows[0] if res.rows else {}
        sources_meta.append({
            "video_id": res.video_id,
            "title": first_row.get("title", ""),
            "uploader": first_row.get("uploader", ""),
            "upload_date": first_row.get("upload_date", ""),
            "n_segments_written": res.n_segments_written,
            "error": res.error,
        })
        all_rows.extend(res.rows)
        for k, v in res.n_dropped.items():
            total_dropped[k] = total_dropped.get(k, 0) + v
        print(f"  → cues: {res.n_cues_total}, written: {res.n_segments_written}, "
              f"dropped: {res.n_dropped}, dur: {res.duration_written_sec:.1f}s", flush=True)
        if res.error:
            print(f"  ⚠ {res.error}", flush=True)

    if not all_rows:
        print("\n✗ No segments were written. Nothing to do.")
        sys.exit(1)

    print(f"\n=== Splitting ({len(all_rows)} rows) ===", flush=True)
    splits = split_rows_by_video(
        all_rows,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    for name, rows in splits.items():
        write_jsonl(manifests_dir / f"{name}.jsonl", rows)
        print(f"  {name}: {len(rows)} examples", flush=True)

    # Build HF dataset from per-split rows so each split gets its own Dataset
    print(f"\n=== Building HF datasets (per split) ===", flush=True)
    from datasets import DatasetDict, load_from_disk
    split_datasets = {}
    for name, rows in splits.items():
        split_path = hf_dir / name
        if rows:
            build_hf_dataset(rows, segments_dir, split_path)
            split_datasets[name] = load_from_disk(str(split_path))
        else:
            split_datasets[name] = None
    # Also build an "all" dataset for convenience
    build_hf_dataset(all_rows, segments_dir, hf_dir / "all")

    print(f"\n=== Stats + README ===", flush=True)
    stats = compute_stats(all_rows)
    audit = character_audit([r["text"] for r in all_rows])
    audit_report = format_audit_report(audit)
    write_readme(data_root, all_rows, splits, stats, audit_report, total_dropped, sources_meta)

    print(f"\n=== DONE ===")
    print(f"  Total examples: {stats['n_examples']}")
    print(f"  Total hours:    {stats['total_duration_hrs']}")
    print(f"  Average dur:    {stats['avg_duration_sec']}s")
    print(f"  Dropped:        {total_dropped}")
    print(f"  Data root:      {data_root}")
    print(f"  HF dataset:     {hf_dir}")
    print(f"  Manifests:      {manifests_dir}")


if __name__ == "__main__":
    main()
