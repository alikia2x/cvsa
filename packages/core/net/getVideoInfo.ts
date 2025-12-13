import networkDelegate from "@core/net/delegate";
import type { VideoInfoData, VideoInfoResponse } from "@core/net/bilibili.d";
import logger from "@core/log";

/*
 * Fetch video metadata from bilibili API
 * @param {number} aid - The video's aid
 * @param {string} task - The task name used in scheduler. It can be one of the following:
 * - snapshotVideo
 * - getVideoInfo
 * - snapshotMilestoneVideo
 * @returns {Promise<VideoInfoData | number>} VideoInfoData or the error code returned by bilibili API
 * @throws {NetSchedulerError} - The error will be thrown in following cases:
 * - No proxy is available currently: with error code `NO_PROXY_AVAILABLE`
 * - The native `fetch` function threw an error: with error code `FETCH_ERROR`
 * - The alicloud-fc threw an error: with error code `ALICLOUD_FC_ERROR`
 */
export async function getVideoInfo(
	aid: number,
	task: "snapshotVideo" | "getVideoInfo" | "snapshotMilestoneVideo"
): Promise<
	| {
			data: VideoInfoData;
			time: number;
	  }
	| number
> {
	const url = `https://api.bilibili.com/x/web-interface/view?aid=${aid}`;
	const { data, time } = await networkDelegate.request<VideoInfoResponse>(url, task);
	const errMessage = `Error fetching metadata for ${aid}:`;
	if (data.code !== 0) {
		logger.error(errMessage + data.code + "-" + data.message, "net", "fn:getVideoInfo");
		return data.code;
	}
	return {
		data: data.data,
		time: time
	};
}
