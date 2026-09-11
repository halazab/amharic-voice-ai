# Amharic Voice AI — ASR Dataset & Pipeline

A complete, production-ready pipeline for building **HuggingFace-compatible Amharic speech recognition datasets** from YouTube videos with Amharic auto-captions. Built to train Whisper / wav2vec2 / XLS-R models for Amharic ASR.

## ✅ Current dataset status

| Metric | Value |
|---|---|
| Source videos processed | 7 |
| Total source video duration | 8.30 hours |
| **Total usable audio** | **7.28 hours (26,207 s)** |
| Total segments | 6,401 |
| Train split | 5,639 examples (6.39 h) |
| Validation split | 407 examples (0.46 h) |
| Test split | 355 examples (0.44 h) |
| Audio format | 16 kHz mono 16-bit PCM WAV |
| Average segment duration | 4.09 s |
| Ge'ez-only text | ✅ Character audit clean |
| Per-video split isolation | ✅ 7 sources, properly split |

## Source videos

| ID | Title | Uploader | Duration | Segments |
|---|---|---|---|---|
| `Lmq9SKgU0BA` | 🔴 Rophnan Interview | Asabiw Inspiration | 11.5 min | 157 |
| `s5dW8J7vDnc` | 📚 የተዋጣለት ተናጋሪ የመሆን ጥበብ (audiobook) | Teddys Podcast | ~5.5 h | 4,811 |
| `gG6pTF1r3e4` | I Tried the BEST Vs. WORST Rated Restaurant in Ethiopia | — | 28 min | 628 |
| `PM0uEIYKtCM` | ሙሉ አማርኛ ፊደላት All Amharic Alphabets | — | 27 min | 407 |
| `Y8d23QZihQw` | ድህነትን እሮጦ ያሸነፈው ሻለቃ ሀይሌ ገብረስላሴ | — | 27 min | 355 |
| `d8-GQtbaAXY` | What can we learn from John? — Pastor Gugssa Biru | — | 27 min | 19 |
| `sMnPHhQaoyM` | ከብ/ጄኔራል ካሳየ ጨመዳ ጋር ተደረገ ቆይታ | — | 29 min | 24 |

## What this repo contains

- **`scripts/`** — End-to-end pipeline code:
  - `amharic_dataset.py` — main pipeline (download → segment → clean → split → HF pack)
  - `amharic_text_utils.py` — Ge'ez-only text cleaner + number-to-words + character audit
  - `find_amharic_videos.py` — search YouTube for Amharic-captioned candidates
  - `verify_candidates.py` — verify each candidate has Amharic captions
  - `start_bgutil.py` — local bgutil-pot server (for YouTube bot-block bypass)
  - `smoke_test.py` — synthetic-data end-to-end test
  - `validate_dataset.py` — load dataset back and inspect samples
  - `rebuild_dataset.py` — rebuild manifests + HF dataset from existing segments
- **`data/`** — The actual dataset (HuggingFace format, ready to `load_from_disk`)
  - `raw/` — original downloads (VTT + info.json + normalized audio per video)
  - `segments/<vid>/0001.wav` + `0001.txt` — per-subtitle audio clips (6,401 total)
  - `manifests/{train,validation,test}.jsonl` — JSONL manifests
  - `hf_dataset/{train,validation,test,all}/` — HuggingFace `save_to_disk` format
  - `README.md` + `character_audit.txt` — full provenance + Unicode audit
- **`docs/`** — Candidate video lists, scaling strategy notes

## Pipeline overview

```
YouTube URL
    │
    ▼  yt-dlp (with cookies + bgutil-pot + deno + EJS solver)
16kHz mono WAV + Amharic VTT
    │
    ▼  VTT parse + Ge'ez-only text cleaner (strips annotations, converts digits → words)
Per-subtitle cue = 1 segment (WAV + TXT)
    │
    ▼  Silence trim ~0.2s lead/tail + duration filter (1-30s)
Cleaned segments
    │
    ▼  Split by source video (not random!) → 90/5/5 train/val/test
JSONL manifests + HuggingFace Dataset
    │
    ▼  save_to_disk() → load_from_disk() ready
Trains Whisper/wav2vec2 out-of-the-box
```

## Quick start

### 1. Install dependencies

```bash
python3 -m pip install \
  yt-dlp yt-dlp-ejs yt-dlp-get-pot bgutil-ytdlp-pot-provider \
  "datasets==2.20.0" soundfile librosa
# Also install deno (https://deno.land) and bun (https://bun.sh)
```

### 2. Get your YouTube cookies (REQUIRED for bot-block bypass)

YouTube now blocks most cloud IPs and requires authentication even for public videos. You need to export cookies from a logged-in browser session:

1. Install the **"Get cookies.txt LOCALLY"** browser extension
2. Open YouTube in your browser, make sure you're signed in
3. Click the extension → "Export" → save as `cookies.txt`

### 3. Start the bgutil-pot server (REQUIRED for PO tokens)

```bash
git clone https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git
cd bgutil-ytdlp-pot-provider/server
bun install
bun run src/main.ts --port 4416 &
```

### 4. Add YouTube URLs

```bash
echo "https://youtu.be/VIDEO_ID_1" > youtube_urls.txt
echo "https://youtu.be/VIDEO_ID_2" >> youtube_urls.txt
# ... etc
```

Or auto-discover Amharic-captioned candidates:

```bash
python3 scripts/find_amharic_videos.py    # searches YouTube, writes candidate_videos.json
python3 scripts/verify_candidates.py       # verifies each has am captions, writes youtube_urls.txt
```

### 5. Run the pipeline

```bash
PATH="$HOME/.deno/bin:$PATH" python3 scripts/amharic_dataset.py \
  --urls youtube_urls.txt \
  --data-root ./data \
  --cookies cookies.txt \
  --min-duration 1.0 --max-duration 30.0 --ideal-min 5.0 --ideal-max 15.0
```

The pipeline is **idempotent** — already-downloaded videos are skipped. You can run it repeatedly to accumulate data.

### 6. Load the dataset

```python
from datasets import load_from_disk

train = load_from_disk("data/hf_dataset/train")
val   = load_from_disk("data/hf_dataset/validation")
test  = load_from_disk("data/hf_dataset/test")

print(f"Train: {len(train)} examples, {sum(train['duration'])/3600:.2f}h")
print(f"Val:   {len(val)} examples, {sum(val['duration'])/3600:.2f}h")
print(f"Test:  {len(test)} examples, {sum(test['duration'])/3600:.2f}h")

print(train[0])
# {'audio': {'path': '0001.wav', 'array': array([..., dtype=float64]), 'sampling_rate': 16000},
#  'text': 'የሰው ልጅ ታላቁ ጥበብ በጉልበቱ የመግዛት ችሎታው ሳይሆን',
#  'duration': 4.83,
#  'source': 'youtube:s5dW8J7vDnc',
#  'speaker': 'unknown',
#  'title': '📚[👉ሙሉ መፅሐፍ]  የተዋጣለት ተናጋሪ የመሆን ጥበብ',
#  'uploader': 'Teddys Podcast',
#  'upload_date': '...',
#  'is_ideal_duration': False}
```

### 7. Train Whisper

```python
from datasets import load_from_disk, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="amharic", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

ds = load_from_disk("data/hf_dataset/train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

ds = ds.map(prepare, remove_columns=ds.column_names)
# ... feed to Trainer
```

## How the text cleaning works

The text cleaner (`amharic_text_utils.py`) enforces:

| Rule | What it does | Example |
|---|---|---|
| Ge'ez script only | Strips Latin letters, emojis, symbols | `"Hello ሰላም"` → dropped |
| Strip annotations | Removes `[ሙዚቃ]`, `(inaudible)`, `>>` speaker markers | `"[ሙዚቃ] ሰላም"` → `"ሰላም"` |
| Strip inline tags | Removes `<i>`, `</i>`, `<00:00:01.500>` VTT timestamps | `"<i>ሰላም</i>"` → `"ሰላም"` |
| Numbers → words | Converts integers 0-9999 to Amharic word form | `"በ 2024"` → `"በ ሁለት ሺህ ሀያ አራት"` |
| Drop large numbers | Numbers >9999 → segment dropped entirely | `"999999"` → segment dropped |
| Keep Ge'ez punct | Preserves `። ፣ ፤ ፥ ፦ ፧ ፨` | unchanged |
| Char audit | Reports every Unicode char in corpus | `character_audit.txt` |

## How the split works

**Splits are by source video, NOT random.** This prevents the same speaker / recording condition from appearing in both train and val/test, which would inflate your scores and hide real-world performance problems.

- **≥3 videos:** Greedy algorithm assigns whole videos to splits based on duration targets. Train 90%, val 5%, test 5%.
- **<3 videos:** Falls back to within-video temporal split (first 90% train, middle 5% val, last 5% test). NOT ideal — warn the user.

## Scaling beyond 7.28 hours

The pipeline is set up to scale. See `docs/SCALING.md` for strategies:

- Run from a residential IP (no bot-block, no cookies needed)
- Process in batches of 5-10 videos per session
- Use multiple Google accounts (cycle cookies.txt files)
- Run locally and `git push` results back

## Quality notes

Auto-captions are **pseudo-labels** with ~5-15% error rate. To produce a sellable model:

1. **Listen to a random 1% sample** by ear
2. **Compute CER, not WER** — Ge'ez words are long compounds, character error rate is the honest metric for Amharic
3. **Have humans correct the worst segments** (cheaper than transcribing from scratch)
4. **Track quality per source video** — drop sources that are consistently bad

## Provenance & license

- Audio sourced from YouTube. Original videos retain their original copyright.
- This dataset is intended for **research and educational ASR training**.
- Each segment's `source` field links back to the original video for provenance and takedown compliance.
- If you redistribute, include this README and the manifest files.

## Repository structure

```
amharic-voice-ai/
├── README.md                       ← this file
├── .gitattributes                  ← LFS rules
├── requirements.txt
├── data/                           ← the dataset (3.3 GB, mostly LFS)
│   ├── README.md                   ← dataset README (auto-generated)
│   ├── character_audit.txt         ← Unicode audit report
│   ├── raw/<vid>/                  ← original audio + VTT + info.json
│   ├── segments/<vid>/0001.wav     ← per-subtitle clips (LFS)
│   ├── manifests/{train,val,test}.jsonl
│   └── hf_dataset/{train,val,test,all}/  ← HuggingFace save_to_disk (LFS)
├── scripts/
│   ├── amharic_dataset.py          ← main pipeline
│   ├── amharic_text_utils.py       ← Ge'ez text cleaner + audit
│   ├── find_amharic_videos.py      ← YouTube search
│   ├── verify_candidates.py        ← caption verification
│   ├── rebuild_dataset.py          ← rebuild from existing segments
│   ├── start_bgutil.py             ← bgutil-pot server launcher
│   ├── smoke_test.py               ← synthetic-data test
│   └── validate_dataset.py         ← load + inspect
└── docs/
    ├── candidate_videos.json       ← search results (192 candidates)
    ├── selected_videos.json
    └── SCALING.md                  ← how to scale to 10+ hours
```
