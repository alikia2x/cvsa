import {VideoListResponse } from "lib/net/bilibili.d.ts";
import logger from "lib/log/logger.ts";
import netScheduler, {NetSchedulerError} from "lib/mq/scheduler.ts";

export async function getLatestVideoAids(page: number = 1, pageSize: number = 10): Promise<number[] | null> {
	const startFrom = 1 + pageSize * (page - 1);
	const endTo = pageSize * page;
	const range = `${startFrom}-${endTo}`
	const errMessage = `Error fetching latest aid for ${range}:`
	try {
		const url = `https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=${pageSize}&pn=${page}`;
		const data = await netScheduler.request<VideoListResponse>(url, 'getLatestVideos');
		if (data.code != 0) {
			logger.error(errMessage + data.message, 'net', 'getLastestVideos');
			return [];
		}
		if (data.data.archives.length === 0) {
			logger.verbose("No more videos found", "net", "getLatestVideos");
			return [];
		}
		return data.data.archives.map(video => video.aid);
	}
	catch (e) {
		const error = e as NetSchedulerError;
		if (error.code == "FETCH_ERROR") {
			const rawError = error.rawError! as Error;
			rawError.message = errMessage + rawError.message;
			logger.error(rawError, 'net', 'getVideoTags');
			return null;
		}
		else {
			// Re-throw the error
			throw e;
		}
	}
}
