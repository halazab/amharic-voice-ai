# Amharic ASR Dataset

Built from YouTube audio + Amharic captions. Generated `2026-09-11 12:17:20`.

## Layout
```
data/
├── raw/                 # original downloads (audio + .vtt + info.json per video)
├── segments/            # cut clips: <vid>/0001.wav + 0001.txt
├── manifests/
│   ├── train.jsonl      # 143 examples (0.167h)
│   ├── validation.jsonl # 7 examples (0.007h)
│   └── test.jsonl       # 7 examples (0.007h)
├── hf_dataset/          # HuggingFace save_to_disk format (load_dataset ready)
├── character_audit.txt  # full unicode audit
└── README.md
```

## Stats
- Total examples: 157
- Total audio: 0.181 hours (651.43s)
- Average clip duration: 4.149s (range 1.309-8.35s)
- Number of source videos: 1
- Number of distinct speakers: 1 (most are "unknown" for YouTube)

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
### `Lmq9SKgU0BA`
- Title: 🔴 ይሄንን ከሰማሁ ቡሃላ ህይወቴ ተቀየረ | Rophnan Interview | Ethiopia| Motivation | Rophnan Music | Rofnan
- Uploader: Asabiw Inspiration
- Upload date: 20250527
- Duration: 689s (11.5 min)
- URL: https://youtu.be/Lmq9SKgU0BA
- Segments written: 157

## Dropped segments (and why)
- too_short: 151
- empty: 4
- too_short_after_trim: 1

## Loading
```python
from datasets import load_from_disk
ds = load_from_disk("data/hf_dataset")
print(ds[0])  # {"audio": {...}, "text": "ሰላም ...", "duration": 8.4, ...}
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
