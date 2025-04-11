import { Job } from "npm:bullmq@5.45.2";
import { db } from "db/init.ts";
import { collectSongs } from "mq/task/collectSongs.ts";

export const collectSongsWorker = async (_job: Job): Promise<void> => {
	const client = await db.connect();
	try {
		await collectSongs(client);
	} finally {
		client.release();
	}
};
