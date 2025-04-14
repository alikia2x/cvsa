import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import logger from "log/logger.ts";

export async function removeAllTimeoutSchedules(client: Client) {
	logger.log(
		"Too many timeout schedules, directly removing these schedules...",
		"mq",
		"fn:scheduleCleanupWorker",
	);
	const query: string = `
			    DELETE FROM snapshot_schedule
				WHERE status IN ('pending', 'processing')
				  AND started_at < NOW() - INTERVAL '30 minutes'
			`;
	await client.queryObject(query);
}