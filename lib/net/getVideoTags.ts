import { VideoTagsResponse } from "lib/net/bilibili.d.ts";
import logger from "lib/log/logger.ts";

export async function getVideoTags(aid: number): Promise<string[] | null> {
    try {
        const url = `https://api.bilibili.com/x/tag/archive/tags?aid=${aid}`;
        const res = await fetch(url);
        const data: VideoTagsResponse = await res.json();
        if (data.code != 0) {
            logger.error(`Error fetching tags for video ${aid}: ${data.message}`, 'net', 'getVideoTags');
            return [];
        }
        return data.data.map((tag) => tag.tag_name);
    }
	catch {
        logger.error(`Error fetching tags for video ${aid}`, 'net', 'getVideoTags');
        return null;
    }
}
