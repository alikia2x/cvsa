import netScheduler from "lib/mq/scheduler.ts";
import { VideoInfoData, VideoInfoResponse } from "lib/net/bilibili.d.ts";
import logger from "lib/log/logger.ts";

export async function getVideoInfo(aid: number, task: string): Promise<VideoInfoData | number> {
	const url = `https://api.bilibili.com/x/web-interface/view?aid=${aid}`;
	const data = await netScheduler.request<VideoInfoResponse>(url, task);
	const errMessage = `Error fetching metadata for ${aid}:`;
	if (data.code !== 0) {
		logger.error(errMessage + data.code + "-" + data.message, "net", "fn:getVideoInfo");
		return data.code;
	}
	return data.data;
}
