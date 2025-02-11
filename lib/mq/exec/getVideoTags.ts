import { Job } from "bullmq";
import { VideoTagsQueue } from "lib/mq/index.ts";
import { DAY, HOUR, MINUTE, SECOND } from "$std/datetime/constants.ts";
import { db } from "lib/db/init.ts";
import { truncate } from "lib/utils/truncate.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import logger from "lib/log/logger.ts";
import { getNullVideoTagsList, updateVideoTags } from "lib/db/allData.ts";
import { getVideoTags } from "lib/net/getVideoTags.ts";
import { NetSchedulerError } from "lib/mq/scheduler.ts";
import { WorkerError } from "src/worker.ts";

const delayMap = [0.5, 3, 5, 15, 30, 60];
const getJobPriority = (diff: number) => {
	let priority;
	if (diff > 14 * DAY) {
		priority = 10;
	} else if (diff > 7 * DAY) {
		priority = 7;
	} else if (diff > DAY) {
		priority = 5;
	} else if (diff > 6 * HOUR) {
		priority = 3;
	} else if (diff > HOUR) {
		priority = 2;
	} else {
		priority = 1;
	}
	return priority;
};

const executeTask = async (client: Client, aid: number, failedCount: number, job: Job) => {
	try {
		const result = await getVideoTags(aid);
		if (!result) {
			failedCount = truncate(failedCount + 1, 0, 5);
			const delay = delayMap[failedCount] * MINUTE;
			logger.log(
				`job:getVideoTags added to queue, delay: ${delayMap[failedCount]} minutes.`,
				"mq",
			);
			await VideoTagsQueue.add("getVideoTags", { aid, failedCount }, { delay, priority: 6 - failedCount });
			return 1;
		}
		await updateVideoTags(client, aid, result);
		logger.log(`Fetched tags for aid: ${aid}`, "task");
		return 0;
	} catch (e) {
		if (!(e instanceof NetSchedulerError)) {
			throw new WorkerError(<Error> e, "task", "getVideoTags/fn:executeTask");
		}
		const err = e as NetSchedulerError;
		if (err.code === "NO_AVAILABLE_PROXY" || err.code === "PROXY_RATE_LIMITED") {
			logger.warn(`No available proxy for fetching tags, delayed. aid: ${aid}`, "task");
			await VideoTagsQueue.add("getVideoTags", { aid, failedCount }, {
				delay: 25 * SECOND * Math.random() + 5 * SECOND,
				priority: job.priority,
			});
			return 2;
		}
		throw new WorkerError(err, "task", "getVideoTags/fn:executeTask");
	}
};

export const getVideoTagsWorker = async (job: Job) => {
	const failedCount = (job.data.failedCount ?? 0) as number;
	const client = await db.connect();
	const aid = job.data.aid;
	if (!aid) {
		return 3;
	}

	const v = await executeTask(client, aid, failedCount, job);
	client.release();
	return v;
};

export const getVideoTagsInitializer = async () => {
	const client = await db.connect();
	const videos = await getNullVideoTagsList(client);
	if (videos.length == 0) {
		return 4;
	}
	const count = await VideoTagsQueue.getJobCounts("wait", "delayed", "active");
	const total = count.delayed + count.active + count.wait;
	const max = 15;
	const rest = truncate(max - total, 0, max);

	let i = 0;
	for (const video of videos) {
		if (i > rest) return 100 + i;
		const aid = video.aid;
		const timestamp = video.published_at;
		const diff = Date.now() - timestamp;
		await VideoTagsQueue.add("getVideoTags", { aid }, { priority: getJobPriority(diff) });
		i++;
	}
	return 0;
};
