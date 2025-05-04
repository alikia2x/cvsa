import { sql } from "@core/db/dbNew";
import { Job } from "bullmq";
import { queueLatestVideos } from "mq/task/queueLatestVideo.ts";

export const getLatestVideosWorker = async (_job: Job): Promise<void> =>{
	await queueLatestVideos(sql);
}
