import os
import re
import torch
import torchaudio
from typing import List
from pypinyin import lazy_pinyin
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from torchaudio.transforms import Resample
from tqdm import tqdm
from utils.ttml import TTMLGenerator
from utils.audio import get_audio_duration

# 初始化设备、模型、分词器、对齐器等
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model().to(device)
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

cc_cedict.load()

def timestamp(seconds: float) -> str:
    """将浮点数秒钟转换为TTML时间戳格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans

def parse_lrc(lrc_file, audio_len):
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
            lrc_data.append((lyric, timestamp))
    
    for i, (lyric, start_time) in enumerate(lrc_data):
        # Skip empty line
        if lyric.strip() == "":
            continue
        if i < len(lrc_data) - 1:
            end_time = lrc_data[i + 1][1]
        else:
            end_time = audio_len
        lrc_data[i] = (lyric, start_time, end_time)
    
    # Filter empty lines again
    lrc_data = [line for line in lrc_data if line[0].strip() != ""]

    return lrc_data

def extract_numbers_from_files(directory):
    """
    读取给定目录，提取文件名中的数字部分，并返回一个包含这些数字的列表。

    :param directory: 目录路径
    :return: 包含数字的列表
    """
    numbers = []
    pattern = re.compile(r'line-(\d+)\.wav')

    try:
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                numbers.append(number)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return None

    return numbers

def process_line(line_idx, start_time):
    with open(f"./temp/lines/line-{line_idx}.txt", "r") as f:
        text = f.read()
        
    waveform, sample_rate = torchaudio.load(f"./temp/lines/line-{line_idx}.wav")

    waveform = waveform[0:1]
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    text_pinyin = lazy_pinyin(text)
    text_normalized = " ".join(text_pinyin)
    
    transcript = text_normalized.split()
    emission, token_spans = compute_alignments(waveform, transcript)
    num_frames = emission.size(1)
    ratio = waveform.size(1) / num_frames

    words = []
    for i in range(len(token_spans)):
        spans = token_spans[i]
        x0 = start_time + int(ratio * spans[0].start) / 16000
        x1 = start_time + int(ratio * spans[-1].end) / 16000
        words.append({
            "word": text[i],
            "start": x0,
            "end": x1
        })
    idx=0
    for item in words:
        if idx == len(words) - 1:
            break
        item["end"] = words[idx + 1]["start"]
        idx+=1
    result = []
    for word in words:
        result.append((word["word"], timestamp(word["start"]), timestamp(word["end"])))
    return result


def align(audio_file: str, lrc_file: str, output_ttml: str, segments_dir: str = "./temp/lines"):
    """
    对齐音频和歌词，并输出TTML文件。
    
    :param audio_file: 音频文件路径
    :param lrc_file: LRC歌词文件路径
    :param output_ttml: 输出TTML文件路径
    :param segments_dir: 存放分割后音频片段的目录，默认为"./segments"
    """
    # 获取音频时长
    duration = get_audio_duration(audio_file)

    # 解析LRC文件
    lrc_data = parse_lrc(lrc_file, duration)

    # 提取要处理的行号
    lines_to_process = sorted(extract_numbers_from_files(segments_dir))

    # 创建TTML生成器实例
    ttml_generator = TTMLGenerator(duration=timestamp(duration))
    
    i = 0
    for line_num in tqdm(lines_to_process):
        start_time = lrc_data[i][1]
        end_time = lrc_data[i][2]
        result = process_line(line_num, start_time)
        ttml_generator.add_lyrics(
            begin=timestamp(start_time), end=timestamp(end_time), agent="v1", itunes_key=f"L{i+1}",
            words=result
        )
        i += 1

    # 保存TTML文件
    ttml_generator.save(output_ttml)
    
if __name__ == "__main__":
    align("./data/1.flac", "./data/1.lrc", "./data/output.ttml", "./temp/lines")