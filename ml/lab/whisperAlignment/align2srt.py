import stable_whisper

def align2srt(lyrics, audio_path, output_path):
    model = stable_whisper.load_model('large-v3')
    result = model.align(audio_path, lyrics, language="Chinese", regroup=False)
    result.to_srt_vtt(output_path, segment_level=False)

AUDIO_FILE = './data/1.mp3'
LYRICS_FILE = './data/1.txt'
OUTPUT_FILE = './data/1.srt'

if __name__ == "__main__":
    with open(LYRICS_FILE, 'r') as f:
        lyrics_content = f.read()
        align2srt(lyrics_content, AUDIO_FILE, OUTPUT_FILE)