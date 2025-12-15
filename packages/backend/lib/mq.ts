import { redis } from "@core/db/redis";
import { type ConnectionOptions, Queue } from "bullmq";

export const LatestVideosQueue = new Queue("latestVideos", {
	connection: redis as ConnectionOptions,
});

export const SnapshotQueue = new Queue("snapshot", {
	connection: redis as ConnectionOptions,
});
