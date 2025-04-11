import { Job } from "npm:bullmq@5.45.2";
import { db } from "db/init.ts";
import { insertVideoInfo } from "mq/task/getVideoDetails.ts";

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
