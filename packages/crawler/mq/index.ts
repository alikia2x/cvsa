import { Queue, ConnectionOptions } from "bullmq";
import { redis } from "@core/db/redis.ts";

export const LatestVideosQueue = new Queue("latestVideos", {
	connection: redis as ConnectionOptions
});

export const ClassifyVideoQueue = new Queue("classifyVideo", {
	connection: redis as ConnectionOptions
});

export const SnapshotQueue = new Queue("snapshot", {
	connection: redis as ConnectionOptions
});
