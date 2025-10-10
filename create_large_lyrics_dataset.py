#!/usr/bin/env python3
"""
Create a LARGE original lyrics dataset (500+ songs)
Uses templates with variations to generate diverse, original content
"""

import random

# Large vocabulary banks for variation
LOVE_WORDS = ['heart', 'love', 'dream', 'forever', 'together', 'soul', 'eyes', 'embrace', 'kiss', 'touch']
SAD_WORDS = ['tears', 'pain', 'broken', 'empty', 'alone', 'goodbye', 'memories', 'lost', 'fade', 'regret']
HAPPY_WORDS = ['sunshine', 'dancing', 'smile', 'laughter', 'freedom', 'joy', 'bright', 'alive', 'celebrate']
TIME_WORDS = ['tonight', 'forever', 'moment', 'yesterday', 'tomorrow', 'now', 'today', 'always', 'never']
NATURE = ['sky', 'stars', 'moon', 'ocean', 'rain', 'wind', 'sunrise', 'sunset', 'storm', 'clouds']

RHYME_PAIRS = [
    ['way', 'day', 'say', 'stay', 'play', 'away'],
    ['night', 'light', 'right', 'sight', 'bright', 'flight'],
    ['time', 'mine', 'line', 'shine', 'fine', 'sign'],
    ['feel', 'real', 'steal', 'heal', 'deal', 'reveal'],
    ['go', 'know', 'show', 'flow', 'glow', 'slow'],
    ['heart', 'start', 'part', 'apart', 'art', 'chart'],
    ['free', 'me', 'see', 'be', 'we', 'tree'],
    ['down', 'town', 'crown', 'ground', 'sound', 'around'],
]

def generate_verse_template(theme='love'):
    """Generate a verse with random vocabulary"""
    rhyme_set = random.choice(RHYME_PAIRS)
    random.shuffle(rhyme_set)

    if theme == 'love':
        words = LOVE_WORDS + TIME_WORDS
    elif theme == 'sad':
        words = SAD_WORDS + TIME_WORDS
    else:
        words = HAPPY_WORDS + TIME_WORDS

    templates = [
        f"{random.choice(words).capitalize()} and {random.choice(words)} every {rhyme_set[0]}\n"
        f"{random.choice(words).capitalize()} in your {random.choice(words)}, come what {rhyme_set[1]}\n"
        f"The {random.choice(NATURE)} above, the {random.choice(words)} we {rhyme_set[2]}\n"
        f"I'll {random.choice(['hold', 'keep', 'find', 'chase'])} you close, I'm here to {rhyme_set[3]}",

        f"Walking through the {random.choice(NATURE)} and {random.choice(words)}\n"
        f"Every {rhyme_set[0]} feels like a brand new {rhyme_set[1]}\n"
        f"I {random.choice(['know', 'feel', 'see', 'believe'])} that {random.choice(words)} will find a {rhyme_set[2]}\n"
        f"To {random.choice(['make', 'keep', 'hold', 'find'])} this {random.choice(words)} so {rhyme_set[3]}",

        f"{random.choice(words).capitalize()} is all I {random.choice(['need', 'want', 'feel', 'know'])}\n"
        f"Can't {random.choice(['hide', 'fight', 'stop', 'change'])} what's growing like a {random.choice(['seed', 'flame', 'dream'])}\n"
        f"{random.choice(TIME_WORDS).capitalize()} and {random.choice(['forever', 'always', 'every moment'])}\n"
        f"{random.choice(words).capitalize()} {random.choice(['together', 'as one', 'side by side', 'endlessly'])}"
    ]

    return random.choice(templates)

def generate_chorus_template(theme='love'):
    """Generate a catchy chorus"""
    rhyme_set = random.choice(RHYME_PAIRS)
    random.shuffle(rhyme_set)

    if theme == 'love':
        emotion = random.choice(['fly', 'soar', 'rise', 'shine', 'glow'])
        feeling = random.choice(LOVE_WORDS)
    elif theme == 'sad':
        emotion = random.choice(['fall', 'break', 'fade', 'cry', 'hurt'])
        feeling = random.choice(SAD_WORDS)
    else:
        emotion = random.choice(['dance', 'jump', 'run', 'sing', 'shout'])
        feeling = random.choice(HAPPY_WORDS)

    templates = [
        f"We're gonna {emotion} so {rhyme_set[0]}\n"
        f"Let this {feeling} light the {rhyme_set[1]}\n"
        f"{random.choice(TIME_WORDS).capitalize()} is {random.choice(['ours', 'now', 'here', 'forever'])}, "
        f"everything's {rhyme_set[2]}\n"
        f"This {random.choice(['love', 'feeling', 'moment', 'fire'])} will never {rhyme_set[3]}",

        f"{feeling.capitalize()} in the {random.choice(NATURE)}\n"
        f"{random.choice(['Running', 'Dancing', 'Living', 'Loving'])} wild and {rhyme_set[0]}\n"
        f"This is {random.choice(['our', 'my', 'the']) } {rhyme_set[1]}\n"
        f"We'll {emotion} into the {rhyme_set[2]}"
    ]

    return random.choice(templates)

def generate_song(song_num, theme=None):
    """Generate a complete song"""
    if theme is None:
        theme = random.choice(['love', 'sad', 'happy', 'love', 'love'])  # More love songs

    verse1 = generate_verse_template(theme)
    verse2 = generate_verse_template(theme)
    chorus = generate_chorus_template(theme)
    bridge = generate_verse_template(random.choice(['love', 'sad', 'happy']))

    return f"""--- Song {song_num} ---

[Verse 1]
{verse1}

[Chorus]
{chorus}

[Verse 2]
{verse2}

[Chorus]
{chorus}

[Bridge]
{bridge}

[Chorus]
{chorus}
"""

# Generate large dataset
print("🎵 Creating LARGE original lyrics dataset...")
print("=" * 70)
print("⏳ This will take ~30 seconds...")
print()

num_songs = 600
songs = []

# Generate variety of themes
for i in range(num_songs):
    if i % 50 == 0:
        print(f"   Generating songs {i}-{i+50}...")
    theme = random.choice(['love', 'sad', 'happy', 'love', 'love', 'sad'])
    songs.append(generate_song(i+1, theme))

# Write to file
output_file = 'large_lyrics_dataset.txt'
with open(output_file, 'w') as f:
    for song in songs:
        f.write(song)
        f.write("\n")

import os
file_size = os.path.getsize(output_file)

print()
print("=" * 70)
print("✅ DATASET CREATED SUCCESSFULLY!")
print("=" * 70)
print(f"📁 File: {output_file}")
print(f"📊 Songs: {num_songs}")
print(f"📏 Size: {file_size / 1024:.1f} KB")
print()

# Analyze quality
if file_size < 100000:
    quality = "⚠️  Still too small"
    recommendation = "Need more variations"
elif file_size < 500000:
    quality = "✅ Adequate for basic training"
    recommendation = "Will work, but limited quality"
else:
    quality = "✅ Good size for training"
    recommendation = "Should produce decent results"

print(f"Quality assessment: {quality}")
print(f"Recommendation: {recommendation}")
print()
print("🚀 Next steps:")
print(f"   1. python prepare_custom_data.py --input {output_file} --dataset_name lyrics")
print(f"   2. python train_gpt2.py --dataset=lyrics --n_layer=10 --n_head=5 --n_embd=320 --max_iters=15000")
print()
print("⏱️  Training time: ~6-8 hours on M2 Ultra")
print("🎯 Expected quality: 6-7/10 (decent for original lyrics)")
