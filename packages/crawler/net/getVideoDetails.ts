import networkDelegate from "@core/net/delegate";
import type { VideoDetailsData, VideoDetailsResponse } from "@core/net/bilibili.d";
import logger from "@core/log";

export async function getVideoDetails(aid: number, archive: boolean = false): Promise<VideoDetailsData | null> {
	const url = `https://api.bilibili.com/x/web-interface/view/detail?aid=${aid}`;
	const data = await networkDelegate.request<VideoDetailsResponse>(url, archive ? "" : "getVideoInfo");
	const errMessage = `Error fetching metadata for ${aid}:`;
	if (data.code !== 0) {
		logger.error(errMessage + data.code + "-" + data.message, "net", "fn:getVideoInfo");
		return null;
	}
	return data.data;
}
