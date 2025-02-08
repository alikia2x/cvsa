import MainQueue from "lib/mq/index.ts";

async function configGetLatestVideos() {
	await MainQueue.add("getLatestVideos", {});
}

export async function initMQ() {
	await configGetLatestVideos()
	console.log("Message queue initialized.")
}
