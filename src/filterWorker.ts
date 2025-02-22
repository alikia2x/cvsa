import { Job, Worker } from "bullmq";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";
import { classifyVideosWorker, classifyVideoWorker } from "lib/mq/exec/classifyVideo.ts";
import { WorkerError } from "lib/mq/schema.ts";
import { lockManager } from "lib/mq/lockManager.ts";

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
	{ connection: redis, concurrency: 4, removeOnComplete: { count: 1000 } },
);

filterWorker.on("active", () => {
	logger.log("Worker (filter) activated.", "mq");
});

filterWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

filterWorker.on("closed", async() => {
	await lockManager.releaseLock("classifyVideos");
})
