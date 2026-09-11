"""Validate the produced HF dataset by loading it back and inspecting samples."""
import json
from pathlib import Path
from datasets import load_from_disk
import soundfile as sf

DATA_ROOT = Path("/home/z/my-project/download/amharic-voice-ai/data")

print("=" * 70)
print("Dataset validation")
print("=" * 70)

# Load each split
splits = {}
for name in ["train", "validation", "test", "all"]:
    p = DATA_ROOT / "hf_dataset" / name
    if p.exists():
        splits[name] = load_from_disk(str(p))
        print(f"  {name:12s}: {len(splits[name])} examples, features: {list(splits[name].features.keys())}")

print()
print("=" * 70)
print("Sample examples from train split")
print("=" * 70)
ds = splits["train"]
for i in [0, 1, 2, 50, 100]:
    if i >= len(ds):
        continue
    ex = ds[i]
    print(f"\n--- Example {i} ---")
    print(f"  audio.path:     {ex['audio']['path']}")
    print(f"  audio.shape:    {ex['audio']['array'].shape}")
    print(f"  audio.sr:       {ex['audio']['sampling_rate']}")
    print(f"  duration:       {ex['duration']}s")
    print(f"  text:           {ex['text']!r}")
    print(f"  source:         {ex['source']}")
    print(f"  speaker:        {ex['speaker']}")
    print(f"  title:          {ex['title'][:80]!r}")
    print(f"  is_ideal_dur:   {ex['is_ideal_duration']}")

print()
print("=" * 70)
print("Manifest files")
print("=" * 70)
for name in ["train", "validation", "test"]:
    p = DATA_ROOT / "manifests" / f"{name}.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            lines = f.readlines()
        print(f"\n{name}.jsonl: {len(lines)} lines")
        if lines:
            ex = json.loads(lines[0])
            print(f"  First entry:")
            for k, v in ex.items():
                if isinstance(v, str) and len(v) > 80:
                    print(f"    {k}: {v[:80]}...")
                else:
                    print(f"    {k}: {v}")

print()
print("=" * 70)
print("Directory structure")
print("=" * 70)
import os
for root, dirs, files in os.walk(DATA_ROOT):
    depth = root.replace(str(DATA_ROOT), "").count(os.sep)
    indent = "  " * depth
    subdir_name = os.path.basename(root) or "data"
    n_files_in_dir = len(files)
    n_segments = len([f for f in files if f.endswith(".wav")])
    print(f"{indent}{subdir_name}/  ({n_files_in_dir} files" + (f", {n_segments} WAVs" if n_segments else "") + ")")
    if depth >= 2:
        # Don't recurse into per-video segment dirs more than 1 level
        dirs[:] = []
    dirs.sort()

print()
print("=" * 70)
print("README.md content")
print("=" * 70)
readme = (DATA_ROOT / "README.md").read_text(encoding="utf-8")
print(readme)

print()
print("=" * 70)
print("Character audit (first 60 lines)")
print("=" * 70)
audit = (DATA_ROOT / "character_audit.txt").read_text(encoding="utf-8")
print("\n".join(audit.split("\n")[:60]))

# Also verify that we can actually load one of the WAVs and check format
print()
print("=" * 70)
print("WAV format verification")
print("=" * 70)
import glob
wavs = sorted(glob.glob(str(DATA_ROOT / "segments" / "*" / "*.wav")))
if wavs:
    info = sf.info(wavs[0])
    print(f"  First WAV: {Path(wavs[0]).name}")
    print(f"    samplerate: {info.samplerate} Hz  (expected 16000)")
    print(f"    channels:   {info.channels}  (expected 1)")
    print(f"    subtype:    {info.subtype}  (expected PCM_16)")
    print(f"    frames:     {info.frames}  ({info.frames/info.samplerate:.2f}s)")
    # Load a different one too
    if len(wavs) > 50:
        info = sf.info(wavs[50])
        print(f"\n  50th WAV: {Path(wavs[50]).name}")
        print(f"    samplerate: {info.samplerate} Hz, channels: {info.channels}, subtype: {info.subtype}")
        print(f"    duration: {info.frames/info.samplerate:.2f}s")
