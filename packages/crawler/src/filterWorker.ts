import { ConnectionOptions, Job, Worker } from "bullmq";
import { redis } from "db/redis.ts";
import logger from "log/logger.ts";
import { classifyVideosWorker, classifyVideoWorker } from "mq/exec/classifyVideo.ts";
import { WorkerError } from "mq/schema.ts";
import { lockManager } from "mq/lockManager.ts";
import Akari from "ml/akari.ts";

Deno.addSignalListener("SIGINT", async () => {
	logger.log("SIGINT Received: Shutting down workers...", "mq");
	await filterWorker.close(true);
	Deno.exit();
});

Deno.addSignalListener("SIGTERM", async () => {
	logger.log("SIGTERM Received: Shutting down workers...", "mq");
	await filterWorker.close(true);
	Deno.exit();
});

await Akari.init();

const filterWorker = new Worker(
	"classifyVideo",
	async (job: Job) => {
		switch (job.name) {
			case "classifyVideo":
				return await classifyVideoWorker(job);
			case "classifyVideos":
				return await classifyVideosWorker();
			default:
				break;
		}
	},
	{ connection: redis as ConnectionOptions, concurrency: 2, removeOnComplete: { count: 1000 } },
);

filterWorker.on("active", () => {
	logger.log("Worker (filter) activated.", "mq");
});

filterWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

filterWorker.on("closed", async () => {
	await lockManager.releaseLock("classifyVideos");
});
