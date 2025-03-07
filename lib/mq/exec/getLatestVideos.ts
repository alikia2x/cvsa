import { Job } from "bullmq";
import { queueLatestVideos } from "lib/mq/task/queueLatestVideo.ts";
import { db } from "lib/db/init.ts";
import { insertVideoInfo } from "lib/mq/task/getVideoInfo.ts";
import { collectSongs } from "lib/mq/task/collectSongs.ts";

export const getLatestVideosWorker = async (_job: Job): Promise<void> => {
	const client = await db.connect();
	try {
		await queueLatestVideos(client);
	} finally {
		client.release();
	}
};

export const collectSongsWorker = async (_job: Job): Promise<void> => {
	const client = await db.connect();
	try {
		await collectSongs(client);
	} finally {
		client.release();
	}
};

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
