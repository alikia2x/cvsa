import { HOUR, MINUTE, SECOND } from "@core/const/time.ts";
import { ClassifyVideoQueue, LatestVideosQueue, SnapshotQueue } from "mq/index.ts";
import logger from "@core/log/logger.ts";
import { initSnapshotWindowCounts } from "db/snapshotSchedule.ts";
import { redis } from "@core/db/redis.ts";
import { sql } from "@core/db/dbNew";

export async function initMQ() {
	await initSnapshotWindowCounts(sql, redis);

	await LatestVideosQueue.upsertJobScheduler("getLatestVideos", {
		every: 1 * MINUTE,
		immediately: true
	});

	await ClassifyVideoQueue.upsertJobScheduler("classifyVideos", {
		every: 5 * MINUTE,
		immediately: true
	});

	await LatestVideosQueue.upsertJobScheduler("collectSongs", {
		every: 3 * MINUTE,
		immediately: true
	});

	await SnapshotQueue.upsertJobScheduler(
		"snapshotTick",
		{
			every: 1 * SECOND,
			immediately: true
		},
		{
			opts: {
				removeOnComplete: 300,
				removeOnFail: 600
			}
		}
	);

	await SnapshotQueue.upsertJobScheduler(
		"bulkSnapshotTick",
		{
			every: 15 * SECOND,
			immediately: true
		},
		{
			opts: {
				removeOnComplete: 60,
				removeOnFail: 600
			}
		}
	);

	await SnapshotQueue.upsertJobScheduler("dispatchMilestoneSnapshots", {
		every: 5 * MINUTE,
		immediately: true
	});

	await SnapshotQueue.upsertJobScheduler("dispatchRegularSnapshots", {
		every: 30 * MINUTE,
		immediately: true
	});

	await SnapshotQueue.upsertJobScheduler("dispatchArchiveSnapshots", {
		every: 2 * HOUR,
		immediately: false
	});

	await SnapshotQueue.upsertJobScheduler("scheduleCleanup", {
		every: 2 * MINUTE,
		immediately: true
	});

	logger.log("Message queue initialized.");
}
