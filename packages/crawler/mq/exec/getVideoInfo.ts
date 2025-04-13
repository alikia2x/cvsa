import { Job } from "npm:bullmq@5.45.2";
import { insertVideoInfo } from "mq/task/getVideoDetails.ts";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import logger from "log/logger.ts";

export const getVideoInfoWorker = async (job: Job): Promise<void> =>
	await withDbConnection<void>(async (client: Client) => {
		const aid = job.data.aid;
		if (!aid) {
			logger.warn("aid does not exists", "mq", "job:getVideoInfo");
			return;
		}
		await insertVideoInfo(client, aid);
	});
