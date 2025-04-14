import { VideoListResponse } from "@core/net/bilibili.d.ts";
import logger from "log/logger.ts";
import networkDelegate from "@core/net/delegate.ts";

export async function getLatestVideoAids(page: number = 1, pageSize: number = 10): Promise<number[]> {
	const startFrom = 1 + pageSize * (page - 1);
	const endTo = pageSize * page;
	const range = `${startFrom}-${endTo}`;
	const errMessage = `Error fetching latest aid for ${range}:`;
	const url = `https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=${pageSize}&pn=${page}`;
	const data = await networkDelegate.request<VideoListResponse>(url, "getLatestVideos");
	if (data.code != 0) {
		logger.error(errMessage + data.message, "net", "getLastestVideos");
		return [];
	}
	if (data.data.archives.length === 0) {
		logger.verbose("No more videos found", "net", "getLatestVideos");
		return [];
	}
	return data.data.archives.map((video) => video.aid);
}
