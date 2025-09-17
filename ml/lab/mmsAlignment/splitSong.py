from pydub import AudioSegment
from utils.cleanTempDir import cleanTempDir
import re

def parse_lrc(lrc_file):
    """解析LRC文件，返回一个包含时间戳和歌词的列表"""
    with open(lrc_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lrc_data = []
    for line in lines:
        # 使用正则表达式匹配时间戳和歌词
        match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            lyric = match.group(3).strip()
            lyric = lyric.replace(" ", "")
            timestamp = minutes * 60 + seconds
            lrc_data.append((timestamp, lyric))
    
    return lrc_data

def split_audio_by_lrc(audio_file, lrc_data):
    """根据LRC数据分割音频文件，并保存为单独的WAV文件"""
    audio = AudioSegment.from_file(audio_file)
    cleanTempDir("./temp/lines")
    
    for i, (start_time, lyric) in enumerate(lrc_data):
        # Skip empty line
        if lyric.strip() == "":
            continue
        if i < len(lrc_data) - 1:
            end_time = lrc_data[i + 1][0]
        else:
            end_time = len(audio) / 1000  # 最后一行歌词到音频结束
        start_time = max(0, start_time - 0.1)  # 前后各扩0.1秒
        end_time = min(len(audio) / 1000, end_time + 0.1)
        start_time_ms = start_time * 1000
        end_time_ms = end_time * 1000
        
        segment = audio[start_time_ms:end_time_ms]
        output_file = f"./temp/lines/line-{i+1}.wav"
        output_script = f"./temp/lines/line-{i+1}.txt"
        output_time = f"./temp/lines/line-{i+1}.time"
        segment.export(output_file, format="wav")
        with open(output_script, "w") as f:
            f.write(lyric)
        with open(output_time, "w") as f:
            f.write(str(start_time)+","+str(end_time))
        print(f"Saved {output_file}")

if __name__ == "__main__":
    lrc_file = "./data/1.lrc"  # LRC文件路径
    audio_file = "./data/1.mp3"  # 音频文件路径
    
    lrc_data = parse_lrc(lrc_file)
    split_audio_by_lrc(audio_file, lrc_data)