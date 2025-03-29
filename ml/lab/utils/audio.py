from pydub import AudioSegment

def get_audio_duration(file_path):
    """
    读取音频文件并获取其时长（秒数）。

    :param file_path: 音频文件的路径
    :return: 音频文件的时长（秒数）
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_in_seconds = len(audio) / 1000.0
        return duration_in_seconds
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None