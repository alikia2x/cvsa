import { VideoListResponse } from "lib/net/bilibili.d.ts";
import formatPublishedAt from "lib/utils/formatTimestampToPostgre.ts";
import { AllDataType } from "lib/db/schema.d.ts";
import logger from "lib/log/logger.ts";

export async function getLatestVideos(page: number = 1, pageSize: number = 10, sleepRate: number = 250, fetchTags: boolean = true): Promise<AllDataType[] | null> {
    try {
        const response = await fetch(`https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=${pageSize}&pn=${page}`);
        const data: VideoListResponse = await response.json();

        if (data.code !== 0) {
            logger.error(`Error fetching videos: ${data.message}`, 'net', 'getLatestVideos');
            return null;
        }

        if (data.data.archives.length === 0) {
            logger.verbose("No more videos found", 'net', 'getLatestVideos');
            return [];
        }

        const videoData = data.data.archives.map((video) => {
            const published_at = formatPublishedAt(video.pubdate + 3600 * 8);
            return {
                aid: video.aid,
                bvid: video.bvid,
                description: video.desc,
                uid: video.owner.mid,
                tags: null,
                title: video.title,
                published_at: published_at,
            } as AllDataType;
        });

        return videoData;
    } catch (error) {
        logger.error(error as Error, "net", "getLatestVideos");
        return null;
    }
}