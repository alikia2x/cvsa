import { Job } from "npm:bullmq@5.45.2";
import { collectSongs } from "mq/task/collectSongs.ts";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

export const collectSongsWorker = (_job: Job): Promise<void> =>
	withDbConnection(async (client: Client) => {
		await collectSongs(client);
	});
