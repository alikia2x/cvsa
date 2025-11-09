import { ConnectionOptions, Job, Worker } from "bullmq";
import {
	archiveSnapshotsWorker,
	bulkSnapshotTickWorker,
	collectSongsWorker,
	dispatchMilestoneSnapshotsWorker,
	dispatchRegularSnapshotsWorker,
	getLatestVideosWorker,
	getVideoInfoWorker,
	scheduleCleanupWorker,
	snapshotTickWorker,
	snapshotVideoWorker,
	takeBulkSnapshotForVideosWorker
} from "mq/exec/executors";
import { redis } from "@core/db/redis";
import logger from "@core/log";
import { lockManager } from "@core/mq/lockManager";
import { WorkerError } from "mq/schema";
import { collectQueueMetrics } from "mq/exec/collectQueueMetrics";
import { directSnapshotWorker } from "mq/exec/directSnapshot";

const releaseLockForJob = async (name: string) => {
	await lockManager.releaseLock(name);
	logger.log(`Released lock: ${name}`, "mq");
};

const releaseAllLocks = async () => {
	const locks = ["dispatchRegularSnapshots", "dispatchArchiveSnapshots", "getLatestVideos"];
	for (const lock of locks) {
		await releaseLockForJob(lock);
	}
};

const shutdown = async (signal: string) => {
	logger.log(`${signal} Received: Shutting down workers...`, "mq");
	await releaseAllLocks();
	await latestVideoWorker.close(true);
	await snapshotWorker.close(true);
	await miscWorker.close(true);
	process.exit(0);
};

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

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
		removeOnFail: { count: 0 }
	}
);

latestVideoWorker.on("active", () => {
	logger.log("Worker (latestVideos) activated.", "mq");
});

latestVideoWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

const snapshotWorker = new Worker(
	"snapshot",
	async (job: Job) => {
		switch (job.name) {
			case "directSnapshot": 
				return await directSnapshotWorker(job);
			case "snapshotVideo":
				return await snapshotVideoWorker(job);
			case "snapshotTick":
				return await snapshotTickWorker(job);
			case "dispatchMilestoneSnapshots":
				return await dispatchMilestoneSnapshotsWorker(job);
			case "dispatchRegularSnapshots":
				return await dispatchRegularSnapshotsWorker(job);
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
	{ connection: redis as ConnectionOptions, concurrency: 50, removeOnComplete: { count: 2000 } }
);

snapshotWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

const miscWorker = new Worker(
	"misc",
	async (job: Job) => {
		switch (job.name) {
			case "collectQueueMetrics":
				return await collectQueueMetrics();
			default:
				break;
		}
	},
	{ connection: redis as ConnectionOptions, concurrency: 5, removeOnComplete: { count: 1000 } }
);

miscWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});