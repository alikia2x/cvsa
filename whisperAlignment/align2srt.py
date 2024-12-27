import stable_whisper

def align2srt(lyrics, audio_path, output_path):
    model = stable_whisper.load_model('large-v3')
    result = model.align(audio_path, lyrics, language="Chinese", regroup=False)
    result.to_srt_vtt(output_path, segment_level=False)