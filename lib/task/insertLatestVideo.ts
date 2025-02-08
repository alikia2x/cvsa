import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getLatestVideos } from "lib/net/getLatestVideos.ts";
import { getLatestVideoTimestampFromAllData, insertIntoAllData, videoExistsInAllData } from "lib/db/allData.ts";
import { sleep } from "lib/utils/sleep.ts";
import { bisectVideoPageInNewList } from "lib/net/bisectVideoStartFrom.ts";

export async function insertLatestVideos(
	client: Client,
	pageSize: number = 10,
	sleepRate: number = 250,
	intervalRate: number = 4000,
): Promise<number | null> {
	const latestVideoTimestamp = await getLatestVideoTimestampFromAllData(client);
	if (latestVideoTimestamp == null) {
		console.error("Cannot get latest video timestamp from current database.");
		return null
	}
	const videoIndex = await bisectVideoPageInNewList(latestVideoTimestamp);
	if (videoIndex == null) {
		console.error("Cannot locate the video through bisect.");
		return null
	}
	let page = Math.floor(videoIndex / pageSize) + 1;
	let failCount = 0;
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
			if (videos.length == 0) {
				console.warn("No more videos found");
				break;
			}
			let allNotExists = true;
			for (const video of videos) {
				const videoExists = await videoExistsInAllData(client, video.aid);
				if (videoExists) {
					allNotExists = false;
				}
				else {
					insertIntoAllData(client, video);
				}
			}
			if (allNotExists) {
				page++;
				console.warn(`All video not exist in the database, going back to older page.`);
				continue;
			}
			console.log(`Page ${page} crawled, total: ${(page - 1) * 20 + videos.length} videos.`);
			page--;
			if (page == 0) {
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
			await sleep(Math.random() * intervalRate + 1000);
		}
	}
	return 0;
}
