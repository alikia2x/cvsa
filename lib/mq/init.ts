import { MINUTE } from "$std/datetime/constants.ts";
import MainQueue from "lib/mq/index.ts";
import logger from "lib/log/logger.ts";

async function configGetLatestVideos() {
	await MainQueue.upsertJobScheduler("getLatestVideos", {
		every: 1 * MINUTE
	})
}

export async function initMQ() {
	await configGetLatestVideos()
	logger.log("Message queue initialized.")
}
