import { Job, Worker } from "bullmq";
import { insertVideosWorker } from "lib/mq/executors.ts";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";

const worker = new Worker(
	"cvsa",
	async (job: Job) => {
		switch (job.name) {
			case "getLatestVideos":
				await insertVideosWorker(job);
				break;
			default:
				break;
		}
	},
	{ connection: redis, concurrency: 4 },
);

worker.on("active", () => {
	logger.log("Worker activated.", "mq");
});

worker.on("error", (err) => {
	logger.error(err);
});
