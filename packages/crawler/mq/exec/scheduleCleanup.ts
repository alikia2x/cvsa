import { Job } from "npm:bullmq@5.45.2";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import logger from "log/logger.ts";
import { scheduleSnapshot, setSnapshotStatus } from "db/snapshotSchedule.ts";
import { SECOND } from "@std/datetime";
import { getTimeoutSchedulesCount } from "mq/task/getTimeoutSchedulesCount.ts";
import { removeAllTimeoutSchedules } from "mq/task/removeAllTimeoutSchedules.ts";

export const scheduleCleanupWorker = async (_job: Job): Promise<void> =>
	await withDbConnection<void>(async (client: Client) => {
		if (await getTimeoutSchedulesCount(client) > 2000) {
			await removeAllTimeoutSchedules(client);
			return;
		}

		const query: string = `
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
		const { rows } = await client.queryObject<{ id: bigint; aid: bigint; type: string }>(query);
		if (rows.length === 0) return;
		for (const row of rows) {
			const id = Number(row.id);
			const aid = Number(row.aid);
			const type = row.type;
			await setSnapshotStatus(client, id, "timeout");
			await scheduleSnapshot(client, aid, type, Date.now() + 10 * SECOND);
			logger.log(
				`Schedule ${id} has not received any response in a while, rescheduled.`,
				"mq",
				"fn:scheduleCleanupWorker",
			);
		}
	}, (e) => {
		logger.error(e as Error, "mq", "fn:scheduleCleanupWorker");
	});
