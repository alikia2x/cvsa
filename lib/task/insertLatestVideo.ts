import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getLatestVideos } from "lib/net/getLatestVideos.ts";
import { getLatestVideoTimestampFromAllData, insertIntoAllData, videoExistsInAllData } from "lib/db/allData.ts";
import { sleep } from "lib/utils/sleep.ts";
import { getVideoPositionInNewList } from "lib/net/bisectVideoStartFrom.ts";
import { SECOND } from "$std/datetime/constants.ts";
import logger from "lib/log/logger.ts";

export async function insertLatestVideos(
	client: Client,
	pageSize: number = 10,
	intervalRate: number = 4000,
): Promise<number | null> {
	const latestVideoTimestamp = await getLatestVideoTimestampFromAllData(client);
	if (latestVideoTimestamp == null) {
		logger.error("Cannot get latest video timestamp from current database.", "net", "fn:insertLatestVideos()");
		return null
	}
	logger.log(`Latest video in the database: ${new Date(latestVideoTimestamp).toISOString()}`, "net", "fn:insertLatestVideos()")
	const videoIndex = await getVideoPositionInNewList(latestVideoTimestamp);
	if (videoIndex == null) {
		logger.error("Cannot locate the video through bisect.", "net", "fn:insertLatestVideos()");
		return null
	}
	if (typeof videoIndex == "object") {
		for (const video of videoIndex) {
			const videoExists = await videoExistsInAllData(client, video.aid);
			if (!videoExists) {
				await insertIntoAllData(client, video);
			}
		}
		return 0;
	}
	let page = Math.floor(videoIndex / pageSize) + 1;
	let failCount = 0;
	const insertedVideos = new Set();
	while (true) {
		try {
			const videos = await getLatestVideos(page, pageSize);
			if (videos == null) {
				failCount++;
				if (failCount > 5) {
					return null;
				}
				continue;
			}
			failCount = 0;
			if (videos.length == 0) {
				logger.verbose("No more videos found", "net", "fn:insertLatestVideos()");
				break;
			}
			for (const video of videos) {
				const videoExists = await videoExistsInAllData(client, video.aid);
				if (!videoExists) {
					await insertIntoAllData(client, video);
					insertedVideos.add(video.aid);
				}
			}
			logger.log(`Page ${page} crawled, total: ${insertedVideos.size} videos.`, "net", "fn:insertLatestVideos()");
			page--;
			if (page < 1) {
				return 0;
			}
		} catch (error) {
			logger.error(error as Error, "net", "fn:insertLatestVideos()");
			failCount++;
			if (failCount > 5) {
				return null;
			}

		} finally {
			await sleep(Math.random() * intervalRate + failCount * 3 * SECOND + SECOND);
		}
	}
	return 0;
}
