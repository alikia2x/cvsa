import logger from "@core/log";
import type { VideoListResponse } from "@core/net/bilibili.d";
import networkDelegate from "@core/net/delegate";

const cacheFile = Bun.file("temp/aidCache.json");
const aidSet = new Set();
try {
	const cache = (await cacheFile.json()) as number[];
	for (const aid of cache) {
		aidSet.add(aid);
	}
} catch {}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function getLatestVideos(page: number = 1, pageSize: number = 50) {
	const url = `https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=${pageSize}&pn=${page}`;
	const { data } = await networkDelegate.request<VideoListResponse>(url, "annualArchive");
	if (data.code !== 0) {
		logger.error(data.message, "net", "getLastestVideos");
		return null;
	}
	return data.data;
}

async function getAidPage(page: number) {
	try {
		const data = await getLatestVideos(page);
		if (!data) {
			return page;
		}
		if (data.archives.length === 0) {
			return -1;
		}
		for (const video of data.archives) {
			aidSet.add(video.aid);
		}
		logger.log(`Fetched page ${page} with ${data.archives.length} videos`);
		return 0;
	} catch (e) {
		logger.error(e as Error, "net", "getAidPage");
		return page;
	}
}

const concurrency = 35;
const groupTime = 3000;

let minPage = 1;
let maxPage = minPage + concurrency - 1;
const tasks = [];
while (true) {
	try {
		const startTime = performance.now();
		if (tasks.length === 0) {
			for (let i = minPage; i <= maxPage; i++) {
				tasks.push(getAidPage(i));
			}
			minPage += concurrency;
			maxPage += concurrency;
		}
		const results: number[] = await Promise.all(tasks);
		const reachEnd = results.some((result) => result === -1);
		if (reachEnd) {
			break;
		}
		const erroredPages = results.filter((result) => result > 0);
		for (const page of erroredPages) {
			tasks.push(getAidPage(page));
		}
		tasks.splice(0, concurrency);
		logger.log(`Processed page ${minPage} to ${maxPage}`);

		const endTime = performance.now();
		const sleepTime = groupTime - (endTime - startTime);
		if (sleepTime > 0) {
			await sleep(sleepTime);
		}
	} catch (e) {
		logger.error(e as Error, "net", "getAllAids");
	} finally {
		await cacheFile.write(JSON.stringify(Array.from(aidSet)));
	}
}

await cacheFile.write(JSON.stringify(Array.from(aidSet)));

process.exit(0);
