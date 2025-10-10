#!/usr/bin/env python3
"""
Create sample song lyrics dataset for training
Generates original lyrics in various styles to avoid copyright issues
"""

import random

def generate_pop_song():
    """Generate a pop-style song"""
    themes = ['love', 'heartbreak', 'summer', 'dancing', 'freedom']
    theme = random.choice(themes)

    verses = {
        'love': [
            "Every time I see your face\nMy heart begins to race\nYou're the one I can't replace\nIn this endless chase",
            "Dancing in the moonlight glow\nFeeling something that I know\nWith you is where I want to go\nLet the feelings flow"
        ],
        'heartbreak': [
            "Empty room and silent phone\nNever felt so alone\nWhat we had is now long gone\nI should have known",
            "Memories fade like morning mist\nThinking of what I have missed\nOur last goodbye, our final kiss\nHow'd it come to this"
        ],
        'summer': [
            "Golden rays and ocean breeze\nLife just flows with perfect ease\nSandy toes and swaying trees\nDays like these",
            "Sunset paints the evening sky\nWarm night as the stars fly by\nNo need to question how or why\nJust you and I"
        ]
    }

    chorus = {
        'love': "We can fly so high tonight\nEverything just feels so right\nHold me close and hold me tight\nIn your light",
        'heartbreak': "How do I forget your name\nNothing feels quite the same\nPlaying our old game\nWho's to blame",
        'summer': "This is living, this is free\nJust the ocean, you and me\nThis is where we're meant to be\nWild and carefree"
    }

    verse_options = verses.get(theme, verses['love'])

    return f"""[Verse 1]
{random.choice(verse_options)}

[Chorus]
{chorus.get(theme, chorus['love'])}

[Verse 2]
{random.choice(verse_options)}

[Chorus]
{chorus.get(theme, chorus['love'])}

[Bridge]
{"Time stands still when you're around" if theme == 'love' else "Maybe someday I'll be fine" if theme == 'heartbreak' else "Living for the here and now"}
{"Don't let me down" if theme == 'love' else "Leave the past behind" if theme == 'heartbreak' else "No tomorrow, just the now"}

[Chorus]
{chorus.get(theme, chorus['love'])}
"""

def generate_rock_song():
    """Generate a rock-style song"""
    return """[Verse 1]
Thunder rolling in the night
Chasing shadows, running from the light
Every step I take feels wrong or right
In this endless fight

[Chorus]
We're breaking free from all the chains
Rising up through all the pain
Nothing left for us to lose
This path we choose

[Verse 2]
Fire burning in my soul
Taking back what they once stole
Never gonna play their role
I'm in control

[Chorus]
We're breaking free from all the chains
Rising up through all the pain
Nothing left for us to lose
This path we choose

[Bridge]
Scream it loud, scream it true
I'm not done, not through
This is my revolution
My resolution

[Chorus]
We're breaking free from all the chains
Rising up through all the pain
Nothing left for us to lose
This path we choose
"""

def generate_country_song():
    """Generate a country-style song"""
    return """[Verse 1]
Dirt road running past the old oak tree
Where we carved our names for the world to see
That pickup truck still runs but barely
Takes me back to memories

[Chorus]
Take me home to where I belong
Where the nights are warm and the love is strong
Simple life and a country song
That's where my heart's been all along

[Verse 2]
Mama's cookin' and daddy's pride
Front porch swing on a summer night
Stars above shining bright and wide
That small town life

[Chorus]
Take me home to where I belong
Where the nights are warm and the love is strong
Simple life and a country song
That's where my heart's been all along

[Bridge]
City lights can't compare
To that sweet country air
That's my home, that's my prayer
I'll meet you there

[Chorus]
Take me home to where I belong
Where the nights are warm and the love is strong
Simple life and a country song
That's where my heart's been all along
"""

def generate_rap_song():
    """Generate a rap/hip-hop style song"""
    return """[Verse 1]
Started from the bottom with a dollar and a dream
Nothing ever easy, nothing's quite as it seems
Working through the struggle, building up my team
Now we're living large, living like a king

[Hook]
We made it through the hard times
Now we're living in the spotlight
Started with nothing, now we got everything
This is our time, this is our life

[Verse 2]
Every single setback made me stronger than before
Every closed window led me to an open door
Haters gonna hate but I just keep wanting more
Rising to the top, that's what I'm aiming for

[Hook]
We made it through the hard times
Now we're living in the spotlight
Started with nothing, now we got everything
This is our time, this is our life

[Verse 3]
Looking back at where I came from, how far I've grown
Turned my dreams into reality, made this path my own
Never giving up, that's the only thing I've known
Now I'm sitting here on my throne

[Hook]
We made it through the hard times
Now we're living in the spotlight
Started with nothing, now we got everything
This is our time, this is our life
"""

# Generate sample dataset
print("🎵 Creating sample lyrics dataset...")
print("=" * 70)

songs = []

# Generate variety of songs
for _ in range(20):
    songs.append(generate_pop_song())

for _ in range(10):
    songs.append(generate_rock_song())

for _ in range(10):
    songs.append(generate_country_song())

for _ in range(10):
    songs.append(generate_rap_song())

# Shuffle
random.shuffle(songs)

# Write to file
output_file = 'sample_lyrics.txt'
with open(output_file, 'w') as f:
    for i, song in enumerate(songs):
        f.write(f"--- Song {i+1} ---\n\n")
        f.write(song)
        f.write("\n\n")

print(f"✅ Created {len(songs)} songs")
print(f"📁 Saved to: {output_file}")
print(f"📏 File size: {len(open(output_file).read()) / 1024:.1f} KB")
print()
print("📊 Genre breakdown:")
print(f"   Pop: 20 songs")
print(f"   Rock: 10 songs")
print(f"   Country: 10 songs")
print(f"   Rap: 10 songs")
print()
print("⚠️  Note: This is a SMALL sample dataset for testing")
print("   For real quality, you need 500-1000+ songs")
print()
print("🚀 Next steps:")
print("   1. Prepare dataset:")
print(f"      python prepare_custom_data.py --input {output_file} --dataset_name lyrics")
print()
print("   2. Train model:")
print("      python train_gpt2.py --dataset=lyrics --n_layer=8 --n_head=8 --n_embd=256 --max_iters=5000")
print()
print("   3. Generate lyrics:")
print('      python generate_interactive.py')
print('      Try prompts like: "[Verse 1]" or "[Chorus]"')
