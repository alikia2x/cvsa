import { ConnectionOptions, Job, Worker } from "bullmq";
import { collectSongsWorker, getLatestVideosWorker } from "lib/mq/executors.ts";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";
import { lockManager } from "lib/mq/lockManager.ts";
import { WorkerError } from "lib/mq/schema.ts";
import { getVideoInfoWorker } from "lib/mq/exec/getLatestVideos.ts";
import {
	collectMilestoneSnapshotsWorker,
	snapshotTickWorker,
	takeSnapshotForVideoWorker,
} from "lib/mq/exec/snapshotTick.ts";

Deno.addSignalListener("SIGINT", async () => {
	logger.log("SIGINT Received: Shutting down workers...", "mq");
	await latestVideoWorker.close(true);
	Deno.exit();
});

Deno.addSignalListener("SIGTERM", async () => {
	logger.log("SIGTERM Received: Shutting down workers...", "mq");
	await latestVideoWorker.close(true);
	Deno.exit();
});

const latestVideoWorker = new Worker(
	"latestVideos",
	async (job: Job) => {
		switch (job.name) {
			case "getLatestVideos":
				await getLatestVideosWorker(job);
				break;
			case "getVideoInfo":
				await getVideoInfoWorker(job);
				break;
			case "collectSongs":
				await collectSongsWorker(job);
				break;
			default:
				break;
		}
	},
	{
		connection: redis as ConnectionOptions,
		concurrency: 6,
		removeOnComplete: { count: 1440 },
		removeOnFail: { count: 0 },
	},
);

latestVideoWorker.on("active", () => {
	logger.log("Worker (latestVideos) activated.", "mq");
});

latestVideoWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

latestVideoWorker.on("closed", async () => {
	await lockManager.releaseLock("getLatestVideos");
});

const snapshotWorker = new Worker(
	"snapshot",
	async (job: Job) => {
		switch (job.name) {
			case "snapshotVideo":
				await takeSnapshotForVideoWorker(job);
				break;
			case "snapshotTick":
				await snapshotTickWorker(job);
				break;
			case "collectMilestoneSnapshots":
				await collectMilestoneSnapshotsWorker(job);
				break;
			default:
				break;
		}
	},
	{ connection: redis as ConnectionOptions, concurrency: 10, removeOnComplete: { count: 2000 } },
);

snapshotWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});
