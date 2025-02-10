import { VideoTagsResponse } from "lib/net/bilibili.d.ts";
import netScheduler, {NetSchedulerError} from "lib/mq/scheduler.ts";
import logger from "lib/log/logger.ts";

/*
 * Fetch the tags for a video
 * @param {number} aid The video's aid
 * @return {Promise<string[] | null>} A promise, which resolves to an array of tags,
 * or null if an `fetch` error occurred
 * @throws {NetSchedulerError} If the request failed.
 */
export async function getVideoTags(aid: number): Promise<string[] | null> {
    try {
        const url = `https://api.bilibili.com/x/tag/archive/tags?aid=${aid}`;
        const data = await netScheduler.request<VideoTagsResponse>(url, 'getVideoTags');
        if (data.code != 0) {
            logger.error(`Error fetching tags for video ${aid}: ${data.message}`, 'net', 'getVideoTags');
            return [];
        }
        return data.data.map((tag) => tag.tag_name);
    }
	catch (e) {
        const error = e as NetSchedulerError;
        if (error.errorCode == "FETCH_ERROR") {
            const rawError = error.rawError! as Error;
            rawError.message = `Error fetching tags for video ${aid}: ` + rawError.message;
            logger.error(rawError, 'net', 'getVideoTags');
            return null;
        }
        else {
            // Re-throw the error
            throw e;
        }
    }
}
