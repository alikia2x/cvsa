import { sql } from "@core/db/dbNew";
import logger from "@core/log";

export async function removeAllTimeoutSchedules() {
	logger.log(
		"Too many timeout schedules, directly removing these schedules...",
		"mq",
		"fn:removeAllTimeoutSchedules"
	);
	return sql`
		WITH deleted AS (
			DELETE FROM snapshot_schedule
			WHERE status IN ('pending', 'processing')
			AND started_at < NOW() - INTERVAL '2 minute'
			RETURNING *
		) 
		SELECT count(*) FROM deleted;
	`;
}
