import logger from "@core/log";
import type { VideoDetailsData, VideoDetailsResponse } from "@core/net/bilibili.d";
import networkDelegate from "@core/net/delegate";

/**
 * Fetch detailed video metadata from bilibili API
 * @param aid The aid of the video
 * @returns The detailed metadata of the video, or null if the video does not exist
 * @throws {NetSchedulerError} The caller would need to handle this error
 */
export async function getVideoDetails(aid: number): Promise<VideoDetailsData | null> {
	const url = `https://api.bilibili.com/x/web-interface/view/detail?aid=${aid}`;
	const { data } = await networkDelegate.request<VideoDetailsResponse>(url, "getVideoInfo");
	const errMessage = `Error fetching metadata for ${aid}:`;
	if (data.code !== 0) {
		logger.error(`${errMessage + data.code}-${data.message}`, "net", "fn:getVideoInfo");
		return null;
	}
	return data.data;
}
