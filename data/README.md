# Amharic ASR Dataset

Built from YouTube audio + Amharic captions. Generated `2026-09-11 13:11:33`.

## Layout
```
data/
├── raw/                 # original downloads (audio + .vtt + info.json per video)
├── segments/            # cut clips: <vid>/0001.wav + 0001.txt
├── manifests/
│   ├── train.jsonl      # 5639 examples (6.385h)
│   ├── validation.jsonl # 407 examples (0.458h)
│   └── test.jsonl       # 355 examples (0.437h)
├── hf_dataset/          # HuggingFace save_to_disk format (load_dataset ready)
├── character_audit.txt  # full unicode audit
└── README.md
```

## Stats
- Total examples: 6401
- Total audio: 7.28 hours (26207.29s)
- Average clip duration: 4.094s (range 1.0-28.85s)
- Number of source videos: 7
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
- URL: https://youtu.be/Lmq9SKgU0BA
- Segments written: 157

### `PM0uEIYKtCM`
- Title: ሙሉ አማርኛ  ፊደላት  All Amharic Alphabets
- Uploader: Ethio ሀሁ ፊደል
- Upload date: 20250917
- URL: https://youtu.be/PM0uEIYKtCM
- Segments written: 407

### `Y8d23QZihQw`
- Title: ድህነትን እሮጦ ያሸነፈው ሻለቃ ሀይሌ ገብረስላሴ
- Uploader: Efi G
- Upload date: 20260515
- URL: https://youtu.be/Y8d23QZihQw
- Segments written: 355

### `d8-GQtbaAXY`
- Title: What can we learn from John? Pastor Gugssa Biru አማርኛ ስብከት Amharic Preaching
- Uploader: Biru Gugssa A
- Upload date: 20121201
- URL: https://youtu.be/d8-GQtbaAXY
- Segments written: 19

### `gG6pTF1r3e4`
- Title: I Tried the BEST Vs. WORST Rated Restaurant in Ethiopia... Day 22 in Ethiopia
- Uploader: KmoneyTooClever
- Upload date: 20250111
- URL: https://youtu.be/gG6pTF1r3e4
- Segments written: 628

### `s5dW8J7vDnc`
- Title: 📚[👉ሙሉ መፅሐፍ]  የተዋጣለት ተናጋሪ የመሆን ጥበብ (ሙሉ የድምጸ መጽሐፍ) | Teddys Podcast
- Uploader: TEDEL TUBE
- Upload date: 20260611
- URL: https://youtu.be/s5dW8J7vDnc
- Segments written: 4811

### `sMnPHhQaoyM`
- Title: ከብ/ጄኔራል ካሳየ ጨመዳ ጋር ተደረገ ቆይታ|
- Uploader: EBC
- Upload date: 20210814
- URL: https://youtu.be/sMnPHhQaoyM
- Segments written: 24

## Dropped segments (and why)
- (none)

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
