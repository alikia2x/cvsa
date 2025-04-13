import { Job } from "bullmq";
import { queueLatestVideos } from "mq/task/queueLatestVideo.ts";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

export const getLatestVideosWorker = (_job: Job): Promise<void> =>
	withDbConnection(async (client: Client) => {
		await queueLatestVideos(client);
	});
