import { MINUTE, SECOND } from "$std/datetime/constants.ts";
import { ClassifyVideoQueue, LatestVideosQueue, SnapshotQueue } from "lib/mq/index.ts";
import logger from "lib/log/logger.ts";
import { initSnapshotWindowCounts } from "lib/db/snapshotSchedule.ts";
import { db } from "lib/db/init.ts";
import { redis } from "lib/db/redis.ts";

export async function initMQ() {
	const client = await db.connect();
	try {
		await initSnapshotWindowCounts(client, redis);

		await LatestVideosQueue.upsertJobScheduler("getLatestVideos", {
			every: 1 * MINUTE,
			immediately: true,
		});

		await ClassifyVideoQueue.upsertJobScheduler("classifyVideos", {
			every: 5 * MINUTE,
			immediately: true,
		});

		await LatestVideosQueue.upsertJobScheduler("collectSongs", {
			every: 3 * MINUTE,
			immediately: true,
		});

		await SnapshotQueue.upsertJobScheduler("snapshotTick", {
			every: 1 * SECOND,
			immediately: true,
		}, {
			opts: {
				removeOnComplete: 1,
				removeOnFail: 1,
			},
		});

		await SnapshotQueue.upsertJobScheduler("collectMilestoneSnapshots", {
			every: 5 * MINUTE,
			immediately: true,
		});

		await SnapshotQueue.upsertJobScheduler("dispatchRegularSnapshots", {
			every: 30 * MINUTE,
			immediately: true,
		});

		await SnapshotQueue.upsertJobScheduler("scheduleCleanup", {
			every: 30 * MINUTE,
			immediately: true,
		});

		logger.log("Message queue initialized.");
	} finally {
		client.release();
	}
}
