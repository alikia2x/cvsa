import { Job } from "bullmq";
import { queueLatestVideos } from "lib/mq/task/queueLatestVideo.ts";
import { db } from "lib/db/init.ts";

export const getLatestVideosWorker = async (_job: Job): Promise<void> => {
	const client = await db.connect();
	try {
		await queueLatestVideos(client);
	} finally {
		client.release();
	}
};
