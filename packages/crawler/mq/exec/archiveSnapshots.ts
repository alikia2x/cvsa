import { Job } from "npm:bullmq@5.45.2";
import { getAllVideosWithoutActiveSnapshotSchedule, scheduleSnapshot } from "db/snapshotSchedule.ts";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import logger from "log/logger.ts";
import { lockManager } from "mq/lockManager.ts";
import { getLatestVideoSnapshot } from "db/snapshot.ts";
import { HOUR, MINUTE } from "$std/datetime/constants.ts";

export const archiveSnapshotsWorker = async (_job: Job) =>
	await withDbConnection<void>(async (client: Client) => {
		const startedAt = Date.now();
		if (await lockManager.isLocked("dispatchArchiveSnapshots")) {
			logger.log("dispatchArchiveSnapshots is already running", "mq");
			return;
		}
		await lockManager.acquireLock("dispatchArchiveSnapshots", 30 * 60);
		const aids = await getAllVideosWithoutActiveSnapshotSchedule(client);
		for (const rawAid of aids) {
			const aid = Number(rawAid);
			const latestSnapshot = await getLatestVideoSnapshot(client, aid);
			const now = Date.now();
			const lastSnapshotedAt = latestSnapshot?.time ?? now;
			const interval = 168;
			logger.log(
				`Scheduled archive snapshot for aid ${aid} in ${interval} hours.`,
				"mq",
				"fn:archiveSnapshotsWorker",
			);
			const targetTime = lastSnapshotedAt + interval * HOUR;
			await scheduleSnapshot(client, aid, "archive", targetTime);
			if (now - startedAt > 250 * MINUTE) {
				return;
			}
		}
	}, (e) => {
		logger.error(e as Error, "mq", "fn:archiveSnapshotsWorker");
	}, async () => {
		await lockManager.releaseLock("dispatchArchiveSnapshots");
	});
