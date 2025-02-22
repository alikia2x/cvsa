import { Job, Worker } from "bullmq";
import { getLatestVideosWorker } from "lib/mq/executors.ts";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";
import { getVideoTagsWorker } from "lib/mq/exec/getVideoTags.ts";
import { getVideoTagsInitializer } from "lib/mq/exec/getVideoTags.ts";
import { lockManager } from "lib/mq/lockManager.ts";
import { WorkerError } from "../lib/mq/schema.ts";

Deno.addSignalListener("SIGINT", async () => {
	logger.log("SIGINT Received: Shutting down workers...", "mq");
	await latestVideoWorker.close(true);
	await videoTagsWorker.close(true);
	Deno.exit();
});

Deno.addSignalListener("SIGTERM", async () => {
	logger.log("SIGTERM Received: Shutting down workers...", "mq");
	await latestVideoWorker.close(true);
	await videoTagsWorker.close(true);
	Deno.exit();
});

const latestVideoWorker = new Worker(
	"latestVideos",
	async (job: Job) => {
		switch (job.name) {
			case "getLatestVideos":
				await getLatestVideosWorker(job);
				break;
			default:
				break;
		}
	},
	{ connection: redis, concurrency: 1, removeOnComplete: { count: 1440 } },
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

const videoTagsWorker = new Worker(
	"videoTags",
	async (job: Job) => {
		switch (job.name) {
			case "getVideoTags":
				return await getVideoTagsWorker(job);
			case "getVideosTags":
				return await getVideoTagsInitializer();
			default:
				break;
		}
	},
	{
		connection: redis,
		concurrency: 6,
		removeOnComplete: {
			count: 1000,
		},
	},
);

videoTagsWorker.on("active", () => {
	logger.log("Worker (videoTags) activated.", "mq");
});

videoTagsWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});
