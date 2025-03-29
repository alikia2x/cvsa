import xml.etree.ElementTree as ET

class TTMLGenerator:
    def __init__(self, duration, xmlns="http://www.w3.org/ns/ttml", xmlns_ttm="http://www.w3.org/ns/ttml#metadata", xmlns_amll="http://www.example.com/ns/amll", xmlns_itunes="http://music.apple.com/lyric-ttml-internal"):
        self.tt = ET.Element("tt", attrib={
            "xmlns": xmlns,
            "xmlns:ttm": xmlns_ttm,
            "xmlns:amll": xmlns_amll,
            "xmlns:itunes": xmlns_itunes
        })
        self.head = ET.SubElement(self.tt, "head")
        self.metadata = ET.SubElement(self.head, "metadata")
        self.body = ET.SubElement(self.tt, "body", attrib={"dur": duration})
        self.div = ET.SubElement(self.body, "div")

    def add_lyrics(self, begin, end, agent, itunes_key, words):
        p = ET.SubElement(self.div, "p", attrib={
            "begin": begin,
            "end": end,
            "ttm:agent": agent,
            "itunes:key": itunes_key
        })
        for word, start, stop in words:
            span = ET.SubElement(p, "span", attrib={"begin": start, "end": stop})
            span.text = word

    def save(self, filename):
        tree = ET.ElementTree(self.tt)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

def extract_lrc_from_ttml(ttml_file):
    def format_time(ttml_time):
        return ttml_time[3:]

    tree = ET.parse(ttml_file)
    root = tree.getroot()
    namespace = {"": "http://www.w3.org/ns/ttml", "ttm": "http://www.w3.org/ns/ttml#metadata"}

    lrc_lines = []
    
    for p in root.findall(".//p", namespace):
        begin = p.attrib.get("begin")
        end = p.attrib.get("end")
        text_content = ""

        for span in p.findall("span", namespace):
            text_content += span.text or ""

        # Format begin and end times
        begin_time = format_time(begin)
        end_time = format_time(end)

        # Add formatted lines to the LRC list
        lrc_lines.append(f"[{begin_time}] {text_content}")
        lrc_lines.append(f"[{end_time}]")  # Add the end time as a separate line

    return "\n".join(lrc_lines)
