import { Job } from "bullmq";
import { collectSongs } from "mq/task/collectSongs";

export const collectSongsWorker = async (_job: Job): Promise<void> => {
	await collectSongs();
	return;
};
