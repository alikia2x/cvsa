import { sql } from "@core/db/dbNew";

export async function getTimeoutSchedulesCount() {
	const rows = await sql<{ count: number }[]>`
		SELECT COUNT(id)
		FROM snapshot_schedule
		WHERE status IN ('pending', 'processing')
			AND started_at < NOW() - INTERVAL '30 minutes'
	`;
	return rows[0].count;
}
