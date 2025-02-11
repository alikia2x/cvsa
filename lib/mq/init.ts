import { MINUTE, SECOND } from "$std/datetime/constants.ts";
import { LatestVideosQueue, VideoTagsQueue } from "lib/mq/index.ts";
import logger from "lib/log/logger.ts";

async function configGetLatestVideos() {
	await LatestVideosQueue.upsertJobScheduler("getLatestVideos", {
		every: 1 * MINUTE,
	});
}

async function configGetVideosTags() {
	await VideoTagsQueue.upsertJobScheduler("getVideosTags", {
		every: 30 * SECOND,
		immediately: true,
	});
}

export async function initMQ() {
	await configGetLatestVideos();
	await configGetVideosTags();
	logger.log("Message queue initialized.");
}
