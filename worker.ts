import { Job, Worker } from "bullmq";
import { getLatestVideosWorker } from "lib/mq/executors.ts";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";

const crawlerWorker = new Worker(
	"cvsa",
	async (job: Job) => {
		switch (job.name) {
			case "getLatestVideos":
				await getLatestVideosWorker(job);
				break;
			default:
				break;
		}
	},
	{ connection: redis, concurrency: 10 },
);

crawlerWorker.on("active", () => {
	logger.log("Worker activated.", "mq");
});

crawlerWorker.on("error", (err) => {
	logger.error(err);
});
