import { MINUTE } from "$std/datetime/constants.ts";
import MainQueue from "lib/mq/index.ts";

async function configGetLatestVideos() {
	await MainQueue.upsertJobScheduler("getLatestVideos", {
		every: 5 * MINUTE
	})
}

export async function initMQ() {
	await configGetLatestVideos()
	console.log("Message queue initialized.")
}
