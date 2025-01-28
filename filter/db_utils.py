import sqlite3
import json

def fetch_entry_data(db_path, aid):
    """
    根据aid从数据库中加载data
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM bili_info_crawl WHERE aid = ?", (aid,))
    fet = cursor.fetchone()
    if fet:
        data = fet[0]
    else:
        data = None
    conn.close()
    return data

def parse_entry_data(data):
    """
    解析JSON数据，提取视频标题、简介、标签、作者简介
    """
    try:
        obj = json.loads(data)
        title = obj["View"]["title"]
        description = obj["View"]["desc"]
        tags = [tag["tag_name"] for tag in obj["Tags"] if tag["tag_type"] in ["old_channel", "topic"]]
        author_info = obj["Card"]["card"]["name"] + ": " + obj["Card"]["card"]["sign"]
        url = "https://www.bilibili.com/video/" + obj["View"]["bvid"]
        return title, description, tags, author_info, url
    except (TypeError, json.JSONDecodeError) as e:
        print(f"Error parsing data: {e}")
        return None, None, None, None, None