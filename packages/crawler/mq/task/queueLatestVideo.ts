import type { Psql } from "@core/db/psql.d";
import { SECOND } from "@core/lib";
import logger from "@core/log";
import { videoExistsInAllData } from "db/bilibili_metadata";
import { LatestVideosQueue } from "mq/index";
import { getLatestVideoAids } from "net/getLatestVideoAids";
import { sleep } from "utils/sleep";

export async function queueLatestVideos(sql: Psql): Promise<number | null> {
	let page = 1;
	let i = 0;
	const videosFound = new Set();
	while (true) {
		const pageSize = page == 1 ? 10 : 30;
		const aids = await getLatestVideoAids(page, pageSize);
		if (aids.length == 0) {
			logger.verbose("No more videos found", "net", "fn:insertLatestVideos()");
			break;
		}
		let allExists = true;
		let delay = 0;
		for (const aid of aids) {
			const videoExists = await videoExistsInAllData(aid);
			if (videoExists) {
				continue;
			}
			await LatestVideosQueue.add(
				"getVideoInfo",
				{ aid },
				{
					delay,
					attempts: 100,
					backoff: {
						type: "fixed",
						delay: SECOND * 5,
					},
				}
			);
			videosFound.add(aid);
			allExists = false;
			delay += Math.random() * SECOND * 1.5;
		}
		i += aids.length;
		logger.log(
			`Page ${page} crawled, total: ${videosFound.size}/${i} videos added/observed.`,
			"net",
			"fn:queueLatestVideos()"
		);
		if (allExists) {
			return 0;
		}
		page++;
		const randomTime = Math.random() * 4000;
		const delta = SECOND;
		await sleep(randomTime + delta);
	}
	return 0;
}
