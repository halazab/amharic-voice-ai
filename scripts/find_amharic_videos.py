"""Search YouTube for Amharic-captioned videos and build a candidate list.

Uses the YouTube Data API v3 (free, no auth needed for public search).
Returns video IDs matching:
  - Has Amharic auto-captions
  - Reasonable duration (3-30 min — sweet spot for interviews/talks)
  - Amharic-related query

We then verify each via yt-dlp that it has am captions before adding to the
final URL list.
"""
import sys
import json
import requests
import time

# YouTube Data API v3 — public search works without OAuth but needs an API key.
# We can use the public "no key" endpoint or use YouTube's discovery doc.
# Actually, the v3 search API requires a key. Let's use a different approach:
# scrape the YouTube search results page directly (it's allowed for personal use).

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Strategy: search YouTube for Amharic-content queries, scrape watch URLs,
# then verify each video has am captions via the bgutil-cookies-yt-dlp setup.

# Use yt-dlp's "ytsearch" extractor for this — it returns search results with metadata.

import yt_dlp

COOKIES = "/home/z/my-project/scripts/cookies.txt"

# Build search queries that target Amharic content
# Use Amharic keywords so we get actual Amharic-spoken videos
QUERIES = [
    "አማርኛ ቃለ ምልምል",        # "Amharic interview"
    "አማርኛ ዜና",              # "Amharic news"
    "አማርኛ ስብከት",            # "Amharic sermon"
    "አማርኛ ንግግር",            # "Amharic conversation"
    "አማርኛ መማሪያ",            # "Amharic lesson"
    "ኢትዮጵያ ዜና",              # "Ethiopia news"
    "አማርኛ ሰበክ",              # "Amharic preaching"
    "amharic interview",
    "amharic news today",
    "amharic motivation",
    "amharic podcast",
    "amharic sermon",
    "amharic conversation",
    "amharic vlog ethiopia",
]

TARGET_HOURS = 12  # we want 12 hours to be safe above 10
AVAILABILITY_RATE = 0.6  # ~60% of search results will have am captions
AVG_VIDEO_MIN = 10  # avg 10 min per video
NEEDED_VIDEOS = int((TARGET_HOURS * 60) / (AVG_VIDEO_MIN * AVAILABILITY_RATE))
# Search for ~2x more candidates than we need
SEARCH_LIMIT = NEEDED_VIDEOS * 2

print(f"Target: {TARGET_HOURS}h of audio")
print(f"Average video: {AVG_VIDEO_MIN}min, with ~{int(AVAILABILITY_RATE*100)}% having am captions")
print(f"Need ~{NEEDED_VIDEOS} videos, will search for {SEARCH_LIMIT} candidates")
print()

# Use yt-dlp's ytsearch to scrape YouTube search results
all_candidates = []
for query in QUERIES:
    print(f"\n=== Searching: {query!r} ===")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "cookiefile": COOKIES,
        "default_search": "ytsearch",
        "playlistend": 30,  # top 30 results per query
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch30:{query}", download=False)
        entries = [e for e in info.get("entries", []) if e]
        print(f"  Found {len(entries)} results")
        for e in entries:
            vid = e.get("id")
            title = e.get("title", "")
            duration = e.get("duration")
            if vid:
                all_candidates.append({
                    "id": vid,
                    "title": title,
                    "duration": duration,
                    "query": query,
                })
    except Exception as e:
        print(f"  search error: {e}")
        continue

# Dedupe by video id
seen = set()
unique_candidates = []
for c in all_candidates:
    if c["id"] not in seen:
        seen.add(c["id"])
        unique_candidates.append(c)

print(f"\n=== Total unique candidates: {len(unique_candidates)} ===")

# Filter by duration — keep 3-30 min videos
filtered = [c for c in unique_candidates if c["duration"] and 180 <= c["duration"] <= 1800]
print(f"After duration filter (3-30 min): {len(filtered)}")

# Save the candidate list
import json
with open("/home/z/my-project/scripts/candidate_videos.json", "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"\nSaved candidates to /home/z/my-project/scripts/candidate_videos.json")

# Print top 20
print("\nTop 20 candidates:")
for c in filtered[:20]:
    print(f"  [{c['id']}] {c['duration']//60}min  {c['title'][:80]}")
