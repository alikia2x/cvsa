import { Job } from "bullmq";
import { collectSongs } from "mq/task/collectSongs.ts";

export const collectSongsWorker = async (_job: Job): Promise<void> =>{
	await collectSongs();
	return;
}