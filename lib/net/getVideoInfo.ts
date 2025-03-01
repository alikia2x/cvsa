import netScheduler from "lib/mq/scheduler.ts";
import { VideoDetailsData, VideoDetailsResponse } from "lib/net/bilibili.d.ts";
import logger from "lib/log/logger.ts";

export async function getVideoInfo(aid: number): Promise<VideoDetailsData | null> {
	const url = `https://api.bilibili.com/x/web-interface/view/detail?aid=${aid}`;
	const data = await netScheduler.request<VideoDetailsResponse>(url, "getVideoInfo");
	const errMessage = `Error fetching metadata for ${aid}:`;
	logger.log("Fetching metadata for " + aid, "net", "fn:getVideoInfo");
	if (data.code !== 0) {
		logger.error(errMessage + data.message, "net", "fn:getVideoInfo");
		return null;
	}
	return data.data;
}
