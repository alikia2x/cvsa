import { Job, Worker } from "bullmq";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";
import { WorkerError } from "src/worker.ts";
import { classifyVideosWorker, classifyVideoWorker } from "lib/mq/exec/classifyVideo.ts";

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
	{ connection: redis, concurrency: 1, removeOnComplete: { count: 1000 } },
);

filterWorker.on("active", () => {
	logger.log("Worker activated.", "mq");
});

filterWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});