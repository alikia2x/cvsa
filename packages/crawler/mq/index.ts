import { type ConnectionOptions, Queue, QueueEventsProducer } from "bullmq";
import { redis } from "bun";

export const LatestVideosQueue = new Queue("latestVideos", {
	connection: redis as ConnectionOptions,
});

export const ClassifyVideoQueue = new Queue("classifyVideo", {
	connection: redis as ConnectionOptions,
});

export const SnapshotQueue = new Queue("snapshot", {
	connection: redis as ConnectionOptions,
});

export const MiscQueue = new Queue("misc", {
	connection: redis as ConnectionOptions,
});

export const latestVideosEventsProducer = new QueueEventsProducer("latestVideos", {
	connection: redis as ConnectionOptions,
});
