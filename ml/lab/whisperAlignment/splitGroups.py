from pydub import AudioSegment
import os

def parse_line(line):
    """Parse a line in the format '1-26|00:42-02:07'."""
    line_range, time_range = line.split('|')
    start_line, end_line = map(int, line_range.split('-'))
    start_time, end_time = time_range.split('-')
    return (start_line, end_line, start_time, end_time)

def time_to_milliseconds(time_str):
    """Convert a time string in HH:MM:SS or MM:SS format to milliseconds."""
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError("Invalid time format")
    return ((hours * 3600 + minutes * 60 + seconds) * 1000)

def split_audio_and_text(mapping_file, audio_file, text_file, output_dir):
    """Split audio and text into corresponding segments based on mapping_file."""
    # Read mapping file
    with open(mapping_file, 'r') as f:
        mappings = [parse_line(line.strip()) for line in f if line.strip()]

    # Load audio file
    audio = AudioSegment.from_file(audio_file)

    # Read text file lines
    with open(text_file, 'r') as f:
        text_lines = f.readlines()

    # Remove empty lines
    text_lines = [line for line in text_lines if line.strip()]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, (start_line, end_line, start_time, end_time) in enumerate(mappings):
        # Extract text segment
        text_segment = text_lines[start_line - 1:end_line]

        # Extract audio segment
        start_ms = time_to_milliseconds(start_time)
        end_ms = time_to_milliseconds(end_time)
        audio_segment = audio[start_ms:end_ms]

        # Save text segment
        text_output_path = os.path.join(output_dir, f'segment_{i + 1}.txt')
        with open(text_output_path, 'w') as text_file:
            text_file.writelines(text_segment)

        # Save audio segment
        audio_output_path = os.path.join(output_dir, f'segment_{i + 1}.mp3')
        audio_segment.export(audio_output_path, format='mp3')

        # Save segment start time
        start_time_output_path = os.path.join(output_dir, f'segment_{i + 1}.start')
        with open(start_time_output_path, 'w') as start_time_file:
            start_time_file.write(str(start_ms / 1000))

        print(f"Saved segment {i + 1}: {text_output_path}, {audio_output_path}")
