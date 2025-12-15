import { redis } from "@core/db/redis";
import logger from "@core/log";
import { lockManager } from "@core/mq/lockManager";
import { type ConnectionOptions, type Job, Worker } from "bullmq";
import Akari from "ml/akari_api";
import { classifyVideosWorker, classifyVideoWorker } from "mq/exec/classifyVideo";
import type { WorkerError } from "mq/schema";

const shutdown = async (signal: string, filterWorker: Worker<any, any, string>) => {
	logger.log(`${signal} Received: Shutting down workers...`, "mq");
	await filterWorker.close(true);
	process.exit(0);
};

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

process.on("SIGINT", () => shutdown("SIGINT", filterWorker));
process.on("SIGTERM", () => shutdown("SIGTERM", filterWorker));

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
