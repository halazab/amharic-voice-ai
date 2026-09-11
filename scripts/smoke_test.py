"""
Smoke-test the FULL amharic_dataset pipeline on synthetic data.

Creates a fake "downloaded" video:
  - raw/<vid>/audio_16k.wav      (silence + 5 short Amharic utterances TTS'd via numpy)
  - raw/<vid>/<vid>.am.vtt        (5 cues with real Amharic text, with noise mixed in)

Then runs process_video() on it to verify that:
  - WAV normalization works
  - VTT parsing works
  - Text cleaning works
  - Per-cue segmentation works
  - JSONL manifests are written correctly
  - HF dataset builds successfully and loads back via load_from_disk
"""
import sys, os, json, shutil
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/home/z/my-project/scripts")
from amharic_dataset import process_video, build_hf_dataset, split_rows_by_video, write_readme, compute_stats
from amharic_text_utils import character_audit, format_audit_report

# Build synthetic test fixture
data_root = Path("/home/z/my-project/scripts/_smoke_test_data")
if data_root.exists():
    shutil.rmtree(data_root)
for sub in ["raw", "segments", "manifests", "hf_dataset"]:
    (data_root / sub).mkdir(parents=True, exist_ok=True)

# 5 cues, 6 seconds each
cues = [
    (0.0, 6.0,   "ሰላም ይህ ሶፊ ሲኒማ ቻናል ነው"),
    (6.5, 12.0,  "እስከዚህ ሰዓት ድረስ አደጋ የደረሰበት ሰው አልነበረም"),
    (12.5, 18.0, "በ 2024 ዓ.ም የተደረገ ግምጫ"),
    (18.5, 25.0, "[ሙዚቃ] ዛሬ እንዴት ናችሁ አደርኩ"),
    (25.5, 32.0, ">> አንድ ሰው እያለ ነበር እና ሌላ ሰውም መለሰለት"),
]

vid = "TEST_VID_001"
raw_vid_dir = data_root / "raw" / vid
raw_vid_dir.mkdir(parents=True, exist_ok=True)

# Create a 33-second 16k mono WAV with some "speech-like" tone bursts at cue positions
sr = 16000
total_samples = int(33.5 * sr)
audio = np.zeros(total_samples, dtype=np.float32)
for start, end, _ in cues:
    s = int(start * sr)
    e = int(end * sr)
    t = np.arange(e - s) / sr
    # Speech-like: amplitude-modulated 220Hz tone (like a vowel), plus noise
    carrier = np.sin(2 * np.pi * 220 * t) * 0.3
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4Hz amplitude modulation (syllable rate)
    noise = np.random.randn(e - s) * 0.05
    audio[s:e] = (carrier * am + noise) * 0.6
# Convert to int16
audio_i16 = (audio * 32767).astype(np.int16)
sf.write(str(raw_vid_dir / "audio_16k.wav"), audio_i16, sr, subtype="PCM_16")

# Write the VTT (use the same format yt-dlp would produce)
vtt_path = raw_vid_dir / f"{vid}.am.vtt"
vtt_lines = ["WEBVTT", ""]
for start, end, text in cues:
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return f"{h:02d}:{m:02d}:{int(s):02d}.{int((s - int(s))*1000):03d}"
    vtt_lines.append(f"{fmt(start)} --> {fmt(end)}")
    vtt_lines.append(text)
    vtt_lines.append("")
vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
print(f"Created synthetic raw fixture: {raw_vid_dir}")
print(f"  audio_16k.wav ({len(audio_i16)/sr:.1f}s @ 16k mono)")
print(f"  {vtt_path.name} ({len(cues)} cues)")

# Now monkey-patch the download function so process_video uses our fixture
import amharic_dataset
orig = amharic_dataset.download_audio_and_subs

def fake_download(url, raw_dir, prefer_lang="am"):
    return {
        "video_id": vid,
        "audio_path": raw_vid_dir / "audio_16k.wav",
        "vtt_path": vtt_path,
        "title": "Synthetic Test Audiobook",
        "uploader": "test_user",
        "upload_date": "20240101",
        "duration": 33.5,
    }
amharic_dataset.download_audio_and_subs = fake_download

# Run process_video
print("\n=== Running process_video ===")
res = process_video(
    f"https://www.youtube.com/watch?v={vid}",
    data_root / "raw",
    data_root / "segments",
    min_duration=1.0,
    max_duration=30.0,
    ideal_min=5.0,
    ideal_max=15.0,
    keep_punctuation=True,
)
print(f"  cues_total:   {res.n_cues_total}")
print(f"  segments_written: {res.n_segments_written}")
print(f"  dropped:      {res.n_dropped}")
print(f"  total_dur:    {res.duration_written_sec:.1f}s")
print(f"  rows:")
for r in res.rows:
    print(f"    - {r['audio']}  dur={r['duration']:.2f}s  text={r['text']!r}")

all_rows = res.rows
sources_meta = [{
    "video_id": vid,
    "title": "Synthetic Test Audiobook",
    "uploader": "test_user",
    "upload_date": "20240101",
    "n_segments_written": res.n_segments_written,
    "error": None,
}]

# Split
print("\n=== Splitting ===")
splits = split_rows_by_video(all_rows, train_frac=0.9, val_frac=0.05, test_frac=0.05)
for name, rows in splits.items():
    print(f"  {name}: {len(rows)} examples")

# Write manifests
from amharic_dataset import write_jsonl
for name, rows in splits.items():
    write_jsonl(data_root / "manifests" / f"{name}.jsonl", rows)

# Build HF dataset
print("\n=== Building HF dataset ===")
build_hf_dataset(all_rows, data_root / "segments", data_root / "hf_dataset")

# Test loading it back
print("\n=== Loading back via load_from_disk ===")
from datasets import load_from_disk
ds = load_from_disk(str(data_root / "hf_dataset"))
print(f"  Dataset: {len(ds)} examples")
print(f"  Features: {ds.features}")
print(f"  First example:")
ex = ds[0]
print(f"    audio:        <Audio at {ex['audio']['path']} sr={ex['audio']['sampling_rate']}>")
print(f"    audio.shape:  {ex['audio']['array'].shape}  dtype={ex['audio']['array'].dtype}")
print(f"    text:         {ex['text']!r}")
print(f"    duration:     {ex['duration']}s")
print(f"    source:       {ex['source']}")
print(f"    speaker:      {ex['speaker']}")

# README + audit
print("\n=== Writing README + audit ===")
stats = compute_stats(all_rows)
audit = character_audit([r["text"] for r in all_rows])
audit_report = format_audit_report(audit)
write_readme(data_root, all_rows, splits, stats, audit_report, {}, sources_meta)
print("  README.md and character_audit.txt written")

# Print manifest contents
print("\n=== train.jsonl contents ===")
with open(data_root / "manifests" / "train.jsonl", encoding="utf-8") as f:
    for line in f:
        print(f"  {line.rstrip()}")

print(f"\n=== ALL DONE — fixture at {data_root} ===")
print("\n=== Character audit (preview) ===")
print(audit_report[:2000])
