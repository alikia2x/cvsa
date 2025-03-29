import netScheduler from "mq/scheduler.ts";
import { VideoDetailsData, VideoDetailsResponse } from "net/bilibili.d.ts";
import logger from "log/logger.ts";

export async function getVideoDetails(aid: number): Promise<VideoDetailsData | null> {
	const url = `https://api.bilibili.com/x/web-interface/view/detail?aid=${aid}`;
	const data = await netScheduler.request<VideoDetailsResponse>(url, "getVideoInfo");
	const errMessage = `Error fetching metadata for ${aid}:`;
	if (data.code !== 0) {
		logger.error(errMessage + data.code + "-" + data.message, "net", "fn:getVideoInfo");
		return null;
	}
	return data.data;
}
