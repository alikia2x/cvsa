import { sql } from "@core/db/dbNew";
import logger from "@core/log/logger.ts";

export async function removeAllTimeoutSchedules() {
	logger.log("Too many timeout schedules, directly removing these schedules...", "mq", "fn:scheduleCleanupWorker");
	return await sql`
		DELETE FROM snapshot_schedule
		WHERE status IN ('pending', 'processing')
		AND started_at < NOW() - INTERVAL '30 minutes'
	`;
}
