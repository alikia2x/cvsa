import { MINUTE, SECOND } from "$std/datetime/constants.ts";
import { ClassifyVideoQueue, LatestVideosQueue, SnapshotQueue } from "lib/mq/index.ts";
import logger from "lib/log/logger.ts";

export async function initMQ() {
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

	await SnapshotQueue.removeJobScheduler('scheduleSnapshotTick');

	logger.log("Message queue initialized.");
}
