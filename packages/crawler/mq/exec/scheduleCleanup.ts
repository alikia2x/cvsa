import { Job } from "bullmq";
import { sql } from "@core/db/dbNew";
import logger from "@core/log/logger.ts";
import { scheduleSnapshot, setSnapshotStatus } from "db/snapshotSchedule.ts";
import { SECOND } from "@core/const/time.ts";
import { getTimeoutSchedulesCount } from "mq/task/getTimeoutSchedulesCount.ts";
import { removeAllTimeoutSchedules } from "mq/task/removeAllTimeoutSchedules.ts";

interface SnapshotSchedule {
	id: bigint;
	aid: bigint;
	type: string;
}

export const scheduleCleanupWorker = async (_job: Job): Promise<void> => {
	try {
		if ((await getTimeoutSchedulesCount()) > 2000) {
			await removeAllTimeoutSchedules();
			return;
		}

		const rows = await sql<SnapshotSchedule[]>`
            SELECT id, aid, type
            FROM snapshot_schedule
            WHERE status IN ('pending', 'processing')
              AND started_at < NOW() - INTERVAL '30 minutes'
            UNION
            SELECT id, aid, type
            FROM snapshot_schedule
            WHERE status IN ('pending', 'processing')
              AND started_at < NOW() - INTERVAL '2 minutes'
              AND type = 'milestone'
		`;

		if (rows.length === 0) return;
		for (const row of rows) {
			const id = Number(row.id);
			const aid = Number(row.aid);
			const type = row.type;
			await setSnapshotStatus(sql, id, "timeout");
			await scheduleSnapshot(sql, aid, type, Date.now() + 10 * SECOND);
			logger.log(
				`Schedule ${id} has not received any response in a while, rescheduled.`,
				"mq",
				"fn:scheduleCleanupWorker"
			);
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:scheduleCleanupWorker");
	}
};
