import { Job } from "bullmq";
import { insertLatestVideos } from "lib/task/insertLatestVideo.ts";
import MainQueue from "lib/mq/index.ts";
import { MINUTE } from "$std/datetime/constants.ts";
import { db } from "lib/db/init.ts";
import { truncate } from "lib/utils/truncate.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import logger from "lib/log/logger.ts";

const delayMap = [5, 10, 15, 30, 60, 60];

const addJobToQueue = (failedCount: number, delay: number) => {
	logger.log(`job:getLatestVideos added to queue, delay: ${(delay / MINUTE).toFixed(2)} minutes.`, "mq");
	MainQueue.upsertJobScheduler("getLatestVideos", {
		every: delay,
	}, {
		data: {
			failedCount: failedCount,
		},
	});
	return;
};

export const insertVideosWorker = async (job: Job) => {
	const failedCount = (job.data.failedCount ?? 0) as number;
	const client = await db.connect();

	try {
		await executeTask(client, failedCount);
	} finally {
		client.release();
	}
	return;
};

const executeTask = async (client: Client, failedCount: number) => {
	logger.log("getLatestVideos now executing", "task");
	const result = await insertLatestVideos(client);
	failedCount = result !== 0 ? truncate(failedCount + 1, 0, 5) : 0;
	if (failedCount !== 0) {
		addJobToQueue(failedCount, delayMap[failedCount] * MINUTE);
	}
	return;
};
