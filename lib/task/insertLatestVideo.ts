import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getLatestVideos } from "lib/net/getLatestVideos.ts";
import { getLatestVideoTimestampFromAllData, insertIntoAllData, videoExistsInAllData } from "lib/db/allData.ts";
import { sleep } from "lib/utils/sleep.ts";
import { getVideoPositionInNewList } from "lib/net/bisectVideoStartFrom.ts";
import { SECOND } from "$std/datetime/constants.ts";

export async function insertLatestVideos(
	client: Client,
	pageSize: number = 10,
	sleepRate: number = 250,
	intervalRate: number = 4000,
): Promise<number | null> {
	const latestVideoTimestamp = await getLatestVideoTimestampFromAllData(client);
	if (latestVideoTimestamp == null) {
		console.error("[func:insertLatestVideos] Cannot get latest video timestamp from current database.");
		return null
	}
	console.log(`[func:insertLatestVideos] Latest video in the database: ${new Date(latestVideoTimestamp).toISOString()}`)
	const videoIndex = await getVideoPositionInNewList(latestVideoTimestamp);
	if (videoIndex == null) {
		console.error("[func:insertLatestVideos] Cannot locate the video through bisect.");
		return null
	}
	if (typeof videoIndex == "object") {
		for (const video of videoIndex) {
			const videoExists = await videoExistsInAllData(client, video.aid);
			if (!videoExists) {
				insertIntoAllData(client, video);
			}
		}
		return 0;
	}
	let page = Math.floor(videoIndex / pageSize) + 1;
	let failCount = 0;
	const insertedVideos = new Set();
	while (true) {
		try {
			const videos = await getLatestVideos(page, pageSize, sleepRate);
			if (videos == null) {
				failCount++;
				if (failCount > 5) {
					return null;
				}
				continue;
			}
			failCount = 0;
			if (videos.length == 0) {
				console.warn("No more videos found");
				break;
			}
			for (const video of videos) {
				const videoExists = await videoExistsInAllData(client, video.aid);
				if (!videoExists) {
					insertIntoAllData(client, video);
					insertedVideos.add(video.aid);
				}
			}
			console.log(`[func:insertLatestVideos] Page ${page} crawled, total: ${insertedVideos.size} videos.`);
			page--;
			if (page < 1) {
				return 0;
			}
		} catch (error) {
			console.error(error);
			failCount++;
			if (failCount > 5) {
				return null;
			}
			continue;
		} finally {
			await sleep(Math.random() * intervalRate + failCount * 3 * SECOND + SECOND);
		}
	}
	return 0;
}
