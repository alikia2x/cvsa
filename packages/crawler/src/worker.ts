import { ConnectionOptions, Job, Worker } from "bullmq";
import { collectSongsWorker, getLatestVideosWorker } from "mq/executors.ts";
import { redis } from "../../core/db/redis.ts";
import logger from "log/logger.ts";
import { lockManager } from "mq/lockManager.ts";
import { WorkerError } from "mq/schema.ts";
import { getVideoInfoWorker } from "mq/exec/getLatestVideos.ts";
import {
	archiveSnapshotsWorker,
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
				return await getLatestVideosWorker(job);
			case "getVideoInfo":
				return await getVideoInfoWorker(job);
			case "collectSongs":
				return await collectSongsWorker(job);
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
				return await takeSnapshotForVideoWorker(job);
			case "snapshotTick":
				return await snapshotTickWorker(job);
			case "collectMilestoneSnapshots":
				return await collectMilestoneSnapshotsWorker(job);
			case "dispatchRegularSnapshots":
				return await regularSnapshotsWorker(job);
			case "scheduleCleanup":
				return await scheduleCleanupWorker(job);
			case "bulkSnapshotVideo":
				return await takeBulkSnapshotForVideosWorker(job);
			case "bulkSnapshotTick":
				return await bulkSnapshotTickWorker(job);
			case "dispatchArchiveSnapshots":
				return await archiveSnapshotsWorker(job);
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
	await lockManager.releaseLock("dispatchArchiveSnapshots");
});
