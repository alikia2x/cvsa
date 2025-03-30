import { ConnectionOptions, Job, Worker } from "bullmq";
import { collectSongsWorker, getLatestVideosWorker } from "mq/executors.ts";
import { redis } from "db/redis.ts";
import logger from "log/logger.ts";
import { lockManager } from "mq/lockManager.ts";
import { WorkerError } from "mq/schema.ts";
import { getVideoInfoWorker } from "mq/exec/getLatestVideos.ts";
import {
	bulkSnapshotTickWorker,
	collectMilestoneSnapshotsWorker,
	regularSnapshotsWorker,
	scheduleCleanupWorker,
	snapshotTickWorker,
	takeBulkSnapshotForVideosWorker,
	takeSnapshotForVideoWorker,
} from "mq/exec/snapshotTick.ts";

Deno.addSignalListener("SIGINT", async () => {
	logger.log("SIGINT Received: Shutting down workers...", "mq");
	await latestVideoWorker.close(true);
	await snapshotWorker.close(true);
	Deno.exit();
});

Deno.addSignalListener("SIGTERM", async () => {
	logger.log("SIGTERM Received: Shutting down workers...", "mq");
	await latestVideoWorker.close(true);
	await snapshotWorker.close(true);
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
			case "dispatchRegularSnapshots":
				await regularSnapshotsWorker(job);
				break;
			case "scheduleCleanup":
				await scheduleCleanupWorker(job);
				break;
			case "bulkSnapshotVideo":
				await takeBulkSnapshotForVideosWorker(job);
				break;
			case "bulkSnapshotTick":
				await bulkSnapshotTickWorker(job);
				break;
			default:
				break;
		}
	},
	{ connection: redis as ConnectionOptions, concurrency: 50, removeOnComplete: { count: 2000 } },
);

snapshotWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

snapshotWorker.on("closed", async () => {
	await lockManager.releaseLock("dispatchRegularSnapshots");
});
