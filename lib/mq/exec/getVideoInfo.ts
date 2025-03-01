import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { insertVideoInfo } from "lib/mq/task/getVideoInfo.ts";

export const getVideoInfoWorker = async (job: Job): Promise<number> => {
	const client = await db.connect();
	try {
		const aid = job.data.aid;
		if (!aid) {
			return 3;
		}
		await insertVideoInfo(client, aid);
		return 0;
	} finally {
		client.release();
	}
};
