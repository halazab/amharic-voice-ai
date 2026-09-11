# Scaling to 10+ hours

## Target math

- Goal: 10 hours of usable audio
- Yield after silence/music removal: ~50% of source video duration
- Need: ~20 hours of source video → ~120 videos at avg 10 min each

## Reality check: YouTube bot-blocking

After approximately 10-15 video metadata requests in rapid succession (5-10 minutes), YouTube will invalidate your session cookies and start returning:

```
[youtube+GetPOT] VIDEO_ID: Sign in to confirm you're not a bot.
```

The bgutil-pot server still generates PO tokens, but YouTube's secondary bot detection (based on request patterns from your cookies) kicks in.

## Strategies to scale

### Strategy 1: Batch processing (recommended for single-user)

Run the pipeline in batches of 10 videos per session:

```bash
# Batch 1 — first 10 URLs
head -10 youtube_urls.txt > batch1.txt
PATH="$HOME/.deno/bin:$PATH" python3 scripts/amharic_dataset.py \
  --urls batch1.txt --data-root ./data --cookies cookies.txt

# Wait 1-2 hours
# Re-export cookies.txt from your browser (session may have rotated)
# Batch 2 — next 10 URLs
sed -n '11,20p' youtube_urls.txt > batch2.txt
PATH="$HOME/.deno/bin:$PATH" python3 scripts/amharic_dataset.py \
  --urls batch2.txt --data-root ./data --cookies cookies_v2.txt

# Continue until you have 120 videos processed
```

The pipeline is **idempotent** — videos already in `data/raw/<vid>/` are skipped.

### Strategy 2: Skip verification, just download

The verifier (`verify_candidates.py`) does a metadata probe that triggers bot detection. Skip it and just feed URLs directly to the main pipeline — failed downloads are skipped gracefully:

```bash
# Take 120 URLs from candidate_videos.json without verifying
python3 -c "
import json
candidates = json.load(open('docs/candidate_videos.json'))
# Sort by duration descending to get more audio per video
candidates.sort(key=lambda c: -(c.get('duration') or 0))
urls = [f'https://youtu.be/{c[\"id\"]}' for c in candidates[:120]]
open('youtube_urls.txt', 'w').write('\n'.join(urls) + '\n')
print(f'Wrote {len(urls)} URLs')
"

PATH="$HOME/.deno/bin:$PATH" python3 scripts/amharic_dataset.py \
  --urls youtube_urls.txt --data-root ./data --cookies cookies.txt
```

Failed videos will log `"No Amharic (am) subtitle available — skipped"` or `"Sign in to confirm you're not a bot"` and move on. Successful downloads accumulate in `data/raw/` and `data/segments/`.

### Strategy 3: Run locally on a residential IP

The bot block is IP-based. If you run from home:

```bash
# On your home machine:
git clone https://github.com/halazab/amharic-voice-ai.git
cd amharic-voice-ai
pip install -r requirements.txt  # see below

# Cookies not strictly needed from residential IP, but recommended
PATH="$HOME/.deno/bin:$PATH" python3 scripts/amharic_dataset.py \
  --urls youtube_urls.txt --data-root ./data

# Then push the new data back to GitHub
git add data/
git commit -m "Add N new videos, total X hours"
git push
```

### Strategy 4: Multiple Google accounts

Cycle through 3-4 different Google accounts (each with its own `cookies.txt`):

```bash
for cookies in cookies_a.txt cookies_b.txt cookies_c.txt cookies_d.txt; do
  PATH="$HOME/.deno/bin:$PATH" python3 scripts/amharic_dataset.py \
    --urls youtube_urls.txt --data-root ./data --cookies "$cookies"
  sleep 600  # 10-min cooldown between accounts
done
```

## Requirements for residential/local setup

Create `requirements.txt`:

```
yt-dlp>=2026.0.0
yt-dlp-ejs>=0.8.0
yt-dlp-get-pot>=0.3.0
bgutil-ytdlp-pot-provider>=2.0.0
datasets==2.20.0
soundfile
librosa
torch (CPU-only is fine)
```

Also install the JS runtimes:

```bash
# Deno (for yt-dlp n-challenge solver)
curl -fsSL https://deno.land/install.sh | sh

# Bun (for bgutil-pot server)
curl -fsSL https://bun.sh/install | bash

# bgutil-pot server
git clone https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git
cd bgutil-ytdlp-pot-provider/server
bun install
```

## Progress tracking

After each batch, check your accumulated audio:

```bash
python3 scripts/validate_dataset.py
# or
python3 -c "
from datasets import load_from_disk
from pathlib import Path
total = 0
for split in ['train', 'validation', 'test']:
    p = Path(f'data/hf_dataset/{split}')
    if p.exists():
        ds = load_from_disk(str(p))
        d = sum(ds['duration'])
        print(f'{split:12s}: {len(ds):4d} examples, {d/3600:.2f} hours')
        total += d
print(f'TOTAL: {total/3600:.2f} hours')
"
```

Stop when you hit 10+ hours.
