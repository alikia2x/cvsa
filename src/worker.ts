import { Job, Worker } from "bullmq";
import { getLatestVideosWorker } from "lib/mq/executors.ts";
import { redis } from "lib/db/redis.ts";
import logger from "lib/log/logger.ts";
import {getVideoTagsWorker} from "lib/mq/exec/getVideoTags.ts";
import { getVideoTagsInitializer } from "lib/mq/exec/getVideoTags.ts";

export class WorkerError extends Error {
	public service?: string;
	public codePath?: string;
	public rawError: Error;
	constructor(rawError: Error, service?: string, codePath?: string) {
		super(rawError.message);
		this.name = "WorkerFailure";
		this.codePath = codePath;
		this.service = service;
		this.rawError = rawError;
	}
}

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
	{ connection: redis, concurrency: 1 },
);

latestVideoWorker.on("active", () => {
	logger.log("Worker activated.", "mq");
});

latestVideoWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
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
	{ connection: redis, concurrency: 6 },
);

videoTagsWorker.on("active", () => {
	logger.log("Worker activated.", "mq");
});

videoTagsWorker.on("error", (err) => {
	const e = err as WorkerError;
	logger.error(e.rawError, e.service, e.codePath);
});

