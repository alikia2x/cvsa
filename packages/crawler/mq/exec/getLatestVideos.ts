import { sql } from "@core/db/dbNew";
import type { Job } from "bullmq";
import { queueLatestVideos } from "mq/task/queueLatestVideo";

export const getLatestVideosWorker = async (_job: Job): Promise<void> => {
	await queueLatestVideos(sql);
};
