import os
from whisperAlignment.splitGroups import split_audio_and_text
from whisperAlignment.align2srt import align2srt
from whisperAlignment.srt2lrc import srt2lrc
from utils.cleanTempDir import cleanTempDir

def alignWithGroup(segments_file, audio_file, lyrics_file, output_file):
    # Clean temp/segments dir (Insure it exists)
    cleanTempDir('./temp/segments')

    # Split groups
    split_audio_and_text(segments_file, audio_file, lyrics_file, 'temp/segments')

    # Get numbers of segments by count "txt" files in temp/segments
    nums = len([name for name in os.listdir('./temp/segments') if name.endswith('.txt')])

    for i in range(1, int(nums) + 1):
        segment_lyric = f"./temp/segments/segment_{str(i)}.txt"
        segment_audio = f"./temp/segments/segment_{str(i)}.mp3"
        segment_srt = f"./temp/segments/segment_{str(i)}.srt"
        segment_lrc = f"./temp/segments/segment_{str(i)}.lrc"
        segment_start = f"./temp/segments/segment_{str(i)}.start"
        with open(segment_lyric, 'r') as f:
            lyrics = f.read()
        align2srt(lyrics, segment_audio, segment_srt)
        with open(segment_start, 'r') as f:
            offset = float(f.read())
        srt2lrc(lyrics, segment_srt, segment_lrc, offset)

    # Combine lrc files
    lrcs = []
    for i in range(1, int(nums) + 1):
        lrcs.append(f"./temp/segments/segment_{str(i)}.lrc")

    with open(output_file, 'w') as f:
        for lrc in lrcs:
            with open(lrc, 'r') as lrc_file:
                f.write(lrc_file.read())
                f.write('\n')

SEGMENTS_FILE = './data/1.group'
AUDIO_FILE = './data/1.mp3'
LYRICS_FILE = './data/1.txt'
OUTPUT_FILE = './data/1.lrc'

if __name__ == "__main__":
    alignWithGroup(SEGMENTS_FILE, AUDIO_FILE, LYRICS_FILE, OUTPUT_FILE)