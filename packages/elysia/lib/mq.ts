import { Queue, ConnectionOptions } from "bullmq";
import { redis } from "@core/db/redis";

export const LatestVideosQueue = new Queue("latestVideos", {
	connection: redis as ConnectionOptions
});
