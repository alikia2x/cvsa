import { VideoListResponse, VideoListVideo } from "lib/net/bilibili.d.ts";
import logger from "lib/log/logger.ts";

export async function getLatestVideos(
	page: number = 1,
	pageSize: number = 10
): Promise<VideoListVideo[] | null> {
	try {
		const response = await fetch(
			`https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=${pageSize}&pn=${page}`,
		);
		const data: VideoListResponse = await response.json();

		if (data.code !== 0) {
			logger.error(`Error fetching videos: ${data.message}`, "net", "getLatestVideos");
			return null;
		}

		if (data.data.archives.length === 0) {
			logger.verbose("No more videos found", "net", "getLatestVideos");
			return [];
		}

		return data.data.archives;
	} catch (error) {
		logger.error(error as Error, "net", "getLatestVideos");
		return null;
	}
}
