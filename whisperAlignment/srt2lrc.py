import pysrt

def parseTime(object):
    return object.hours * 3600 + object.minutes * 60 + object.seconds + object.milliseconds / 1000

def serializeTime(time):
    minutes = int(time / 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def srt2lrc(lyrics, srt_file, lrc_file, time_offset=0):
    subs = pysrt.open(srt_file, encoding='utf-8')

    # 加载歌词并按行分割
    lyrics_lines = lyrics.splitlines()
    
    # 初始化
    aligned_lines = []
    current_line = ""
    start_time = None

    # 遍历 SRT 的每一项
    for sub in subs:
        word = sub.text.strip()
        if not current_line:
            start_time = parseTime(sub.start)  # 记录行的开始时间

        current_line += word

        # 如果当前行匹配到歌词中的一行
        if lyrics_lines and current_line == lyrics_lines[0]:
            end_time = parseTime(sub.end)  # 记录行的结束时间
            aligned_lines.append(f"[{serializeTime(start_time+time_offset)}] {current_line}\n[{serializeTime(end_time+time_offset)}]")

            # 移除已匹配的歌词行并重置
            lyrics_lines.pop(0)
            current_line = ""
            start_time = None
    
    result = []
    # 后处理，只留下最后一行的结束时间
    for i in range(len(aligned_lines) - 1):
        result.append(aligned_lines[i].split('\n')[0])
    result.append(aligned_lines[-1])

    with open(lrc_file, 'w') as f:
        f.write('\n'.join(result))
