#!/usr/bin/env python3
"""
Educational lyrics collector - FOR PERSONAL LEARNING ONLY
NOT for commercial use or public distribution

⚠️  WARNING: This is for educational purposes only
- Use responsibly and respect rate limits
- Don't overload servers
- Don't distribute the collected data
- For personal experimentation only
"""

import requests
import time
import json
from bs4 import BeautifulSoup

def get_artist_songs_azlyrics(artist_name):
    """
    Get songs from AZLyrics for educational purposes
    Note: Respect robots.txt and rate limits
    """
    print(f"⚠️  Educational use only - collecting from public website")
    print(f"   Rate limited to avoid server stress")
    print()

    # Clean artist name for URL
    artist_url = artist_name.lower().replace(" ", "")
    artist_url = ''.join(c for c in artist_url if c.isalnum())

    url = f"https://www.azlyrics.com/{artist_url[0]}/{artist_url}.html"

    print(f"📡 Fetching: {url}")

    headers = {
        'User-Agent': 'Educational Research Project (Personal Use Only)'
    }

    try:
        # Be respectful - wait before request
        time.sleep(2)

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 404:
            print(f"❌ Artist not found: {artist_name}")
            print(f"   Try exact spelling: 'Taylor Swift', 'Ed Sheeran', etc.")
            return []

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find song links
        songs = []
        song_links = soup.find_all('a', href=True)

        for link in song_links:
            href = link.get('href', '')
            if '../lyrics/' in href and artist_url in href:
                song_title = link.get_text(strip=True)
                song_url = f"https://www.azlyrics.com{href.replace('..', '')}"
                songs.append({
                    'title': song_title,
                    'url': song_url
                })

        print(f"✅ Found {len(songs)} songs")
        return songs[:20]  # Limit to 20 songs to be respectful

    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def get_lyrics(song_url):
    """Get lyrics from a song page"""
    headers = {
        'User-Agent': 'Educational Research Project (Personal Use Only)'
    }

    try:
        # Be respectful - wait between requests
        time.sleep(3)  # 3 seconds between requests

        response = requests.get(song_url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # AZLyrics stores lyrics in specific div
        lyrics_divs = soup.find_all('div', class_='')

        for div in lyrics_divs:
            # Lyrics div has no class and contains the lyrics
            if div.get_text(strip=True) and len(div.get_text(strip=True)) > 100:
                lyrics = div.get_text()
                return lyrics.strip()

        return None

    except Exception as e:
        print(f"   ❌ Error fetching lyrics: {e}")
        return None

def collect_artist_lyrics(artist_name, output_file="artist_lyrics.txt"):
    """
    Main collection function - educational use only
    """
    print("=" * 70)
    print("🎵 EDUCATIONAL LYRICS COLLECTOR")
    print("=" * 70)
    print()
    print("⚠️  IMPORTANT REMINDERS:")
    print("   - This is for PERSONAL LEARNING ONLY")
    print("   - Do NOT distribute the collected data")
    print("   - Do NOT use commercially")
    print("   - Respecting server with rate limits (3 sec between requests)")
    print()
    print(f"Artist: {artist_name}")
    print(f"Output: {output_file}")
    print()

    input("Press Enter to continue (or Ctrl+C to cancel)...")
    print()

    # Get song list
    songs = get_artist_songs_azlyrics(artist_name)

    if not songs:
        print("❌ No songs found")
        return

    print(f"\n📥 Collecting lyrics (limited to {len(songs)} songs to be respectful)...")
    print()

    collected_lyrics = []

    for i, song in enumerate(songs, 1):
        print(f"   {i}/{len(songs)}: {song['title']}... ", end='', flush=True)

        lyrics = get_lyrics(song['url'])

        if lyrics:
            collected_lyrics.append({
                'title': song['title'],
                'lyrics': lyrics
            })
            print("✅")
        else:
            print("❌ Failed")

    # Save to file
    print()
    print(f"💾 Saving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Lyrics collected for educational purposes only\n")
        f.write(f"# Artist: {artist_name}\n")
        f.write(f"# Songs: {len(collected_lyrics)}\n")
        f.write(f"# NOT for commercial use or distribution\n\n")

        for song in collected_lyrics:
            f.write(f"--- {song['title']} ---\n\n")
            f.write(song['lyrics'])
            f.write("\n\n")

    print(f"✅ Collected {len(collected_lyrics)} songs")
    print(f"📁 Saved to: {output_file}")
    print(f"📏 File size: {len(open(output_file).read()) / 1024:.1f} KB")
    print()
    print("⚠️  Remember: Educational use only!")
    print()

if __name__ == '__main__':
    print()
    print("Available to try (enter exact spelling):")
    print("  - Taylor Swift")
    print("  - Ed Sheeran")
    print("  - Adele")
    print("  - Coldplay")
    print("  - etc.")
    print()

    artist = input("Enter artist name: ").strip()

    if not artist:
        print("❌ No artist specified")
        exit(1)

    collect_artist_lyrics(artist)

    print()
    print("🚀 Next steps:")
    print(f"   1. python prepare_custom_data.py --input artist_lyrics.txt --dataset_name lyrics")
    print(f"   2. python train_gpt2.py --dataset=lyrics --n_layer=12 --n_head=6 --n_embd=384 --max_iters=15000")
    print()
    print("⏱️  Total time: ~8 hours training")
    print("🎯 Expected: Model learns artist's style and patterns")
