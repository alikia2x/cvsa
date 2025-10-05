import { Job } from "bullmq";
import { insertVideoInfo } from "mq/task/getVideoDetails";
import logger from "@core/log";
import { sql } from "@core/db/dbNew";

export const getVideoInfoWorker = async (job: Job): Promise<void> => {
	const aid = job.data.aid;
	if (!aid) {
		logger.warn("aid does not exists", "mq", "job:getVideoInfo");
		return;
	}
	await insertVideoInfo(sql, aid);
};
