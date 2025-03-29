from utils.ttml import extract_lrc_from_ttml

lrc_output = extract_lrc_from_ttml('./data/1.ttml')

with open('./data/1-final.lrc', 'w') as f:
    f.write(lrc_output)