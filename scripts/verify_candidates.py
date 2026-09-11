"""Verify candidate videos have Amharic captions, and select enough to fill 12h."""
import sys, json, time
import yt_dlp
from pathlib import Path

COOKIES = "/home/z/my-project/scripts/cookies.txt"

# Load candidates
candidates = json.loads(Path("/home/z/my-project/scripts/candidate_videos.json").read_text(encoding="utf-8"))
print(f"Loaded {len(candidates)} candidates")

# Already-processed videos from previous runs (so we don't redo them)
existing_ids = set()
data_root = Path("/home/z/my-project/download/amharic-voice-ai/data")
if (data_root / "raw").exists():
    for v in (data_root / "raw").iterdir():
        if v.is_dir() and len(list(v.iterdir())) > 0:
            existing_ids.add(v.name)
print(f"Already processed: {len(existing_ids)} videos")
if existing_ids:
    print(f"  IDs: {list(existing_ids)[:5]}")

# Skip already-processed
candidates = [c for c in candidates if c["id"] not in existing_ids]
print(f"To verify: {len(candidates)}")

TARGET_HOURS = 12  # we want 12h of source video to be safe
TARGET_SECONDS = TARGET_HOURS * 3600

# Quick probe: just check if Amharic captions exist + get duration
# Use the most permissive client config that worked before
ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "cookiefile": COOKIES,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["am", "am-orig"],
}

verified = []
total_dur_so_far = sum(c["duration"] or 0 for c in existing_ids and [] or candidates[:0])
# Add already-processed durations
for vid in existing_ids:
    info_files = list((data_root / "raw" / vid).glob("*.info.json"))
    if info_files:
        try:
            meta = json.loads(info_files[0].read_text(encoding="utf-8"))
            total_dur_so_far += meta.get("duration", 0) or 0
        except Exception:
            pass
print(f"Starting with {total_dur_so_far/3600:.2f}h already downloaded")

# Iterate candidates
selected = []
cumulative = total_dur_so_far
start_time = time.time()
for i, c in enumerate(candidates):
    if cumulative >= TARGET_SECONDS:
        print(f"\n=== TARGET REACHED: {cumulative/3600:.2f}h ===")
        break
    if (i+1) % 10 == 0:
        print(f"  progress: {i+1}/{len(candidates)} verified, {cumulative/3600:.2f}h accumulated", flush=True)
    vid = c["id"]
    url = f"https://youtu.be/{vid}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        auto = info.get("automatic_captions", {}) or {}
        if "am" not in auto:
            continue
        duration = info.get("duration", 0) or 0
        title = info.get("title", "")
        uploader = info.get("uploader", "")
        upload_date = info.get("upload_date", "")
        # Skip super-short
        if duration < 120:
            continue
        selected.append({
            "id": vid,
            "url": url,
            "title": title,
            "uploader": uploader,
            "upload_date": upload_date,
            "duration": duration,
        })
        cumulative += duration
        print(f"  ✓ {vid}  {duration//60:>3}min  total={cumulative/3600:.2f}h  {title[:60]}", flush=True)
    except Exception as e:
        # Skip on error
        continue

print(f"\n=== Selection complete ===")
print(f"  Total videos selected: {len(selected)}")
print(f"  Total source duration: {cumulative/3600:.2f}h")
print(f"  Time spent: {(time.time()-start_time)/60:.1f} min")

# Add already-processed videos to the front of the list
existing_list = []
for vid in existing_ids:
    info_files = list((data_root / "raw" / vid).glob("*.info.json"))
    if info_files:
        try:
            meta = json.loads(info_files[0].read_text(encoding="utf-8"))
            existing_list.append({
                "id": vid,
                "url": f"https://youtu.be/{vid}",
                "title": meta.get("title", ""),
                "uploader": meta.get("uploader", ""),
                "upload_date": meta.get("upload_date", ""),
                "duration": meta.get("duration", 0) or 0,
            })
        except Exception:
            pass

all_videos = existing_list + selected
print(f"  Combined with existing: {len(all_videos)} videos, {sum(v['duration'] for v in all_videos)/3600:.2f}h")

# Save URL list
with open("/home/z/my-project/scripts/youtube_urls.txt", "w", encoding="utf-8") as f:
    for v in all_videos:
        f.write(v["url"] + "\n")

# Save full metadata
with open("/home/z/my-project/scripts/selected_videos.json", "w", encoding="utf-8") as f:
    json.dump(all_videos, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(all_videos)} URLs to youtube_urls.txt")
print(f"Saved metadata to selected_videos.json")
