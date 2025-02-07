import { VideoTagsResponse } from "lib/net/bilibili.d.ts";

export async function getVideoTags(aid: number): Promise<string[] | null> {
    try {
        const url = `https://api.bilibili.com/x/tag/archive/tags?aid=${aid}`;
        const res = await fetch(url);
        const data: VideoTagsResponse = await res.json();
        if (data.code != 0) {
            console.error(`Error fetching tags for video ${aid}: ${data.message}`);
            return [];
        }
        return data.data.map((tag) => tag.tag_name);
    }
	catch {
        console.error(`Error fetching tags for video ${aid}`);
        return null;
    }
}
