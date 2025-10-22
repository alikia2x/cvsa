import { ConnectionOptions, Job, Worker } from "bullmq";
import { redis } from "bun";
import logger from "@core/log";
import { classifyVideosWorker, classifyVideoWorker } from "mq/exec/classifyVideo";
import { WorkerError } from "mq/schema";
import { lockManager } from "@core/mq/lockManager";
import Akari from "ml/akari";

const shutdown = async (signal: string) => {
	logger.log(`${signal} Received: Shutting down workers...`, "mq");
	await filterWorker.close(true);
	process.exit(0);
};

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

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
	{ connection: redis as ConnectionOptions, concurrency: 2, removeOnComplete: { count: 1000 } }
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
