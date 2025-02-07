import { VideoListResponse } from "lib/net/bilibili.d.ts";
import formatPublishedAt from "lib/utils/formatTimestampToPostgre.ts";
import { getVideoTags } from "lib/net/getVideoTags.ts";
import { AllDataType } from "lib/db/schema.d.ts";
import { sleep } from "lib/utils/sleep.ts";

export async function getLatestVideos(page: number = 1, pageSize: number = 10): Promise<AllDataType[] | null> {
    try {
        const response = await fetch(`https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=${pageSize}&pn=${page}`);
        const data: VideoListResponse = await response.json();

        if (data.code !== 0) {
            console.error(`Error fetching videos: ${data.message}`);
            return null;
        }

        if (data.data.archives.length === 0) {
            console.warn("No more videos found");
            return [];
        }

        const videoPromises = data.data.archives.map(async (video) => {
            const published_at = formatPublishedAt(video.pubdate + 3600 * 8);
			sleep(Math.random() * pageSize * 250);
            const tags = await getVideoTags(video.aid);
			let processedTags = null;
			if (tags !== null) {
				processedTags = tags.join(',');
			}
            return {
                aid: video.aid,
                bvid: video.bvid,
                description: video.desc,
                uid: video.owner.mid,
                tags: processedTags,
                title: video.title,
                published_at: published_at,
            } as AllDataType;
        });

        const result = await Promise.all(videoPromises);

        return result;
    } catch (error) {
        console.error(error);
        return null;
    }
}