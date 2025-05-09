import { Job } from "bullmq";
import { getAllVideosWithoutActiveSnapshotSchedule, scheduleSnapshot } from "db/snapshotSchedule.ts";
import logger from "@core/log/logger.ts";
import { lockManager } from "mq/lockManager.ts";
import { getLatestVideoSnapshot } from "db/snapshot.ts";
import { HOUR, MINUTE } from "@core/const/time.ts";
import { sql } from "@core/db/dbNew";

export const archiveSnapshotsWorker = async (_job: Job) => {
	try {
		const startedAt = Date.now();
		if (await lockManager.isLocked("dispatchArchiveSnapshots")) {
			logger.log("dispatchArchiveSnapshots is already running", "mq");
			return;
		}
		await lockManager.acquireLock("dispatchArchiveSnapshots", 30 * 60);
		const aids = await getAllVideosWithoutActiveSnapshotSchedule(sql);
		for (const rawAid of aids) {
			const aid = Number(rawAid);
			const latestSnapshot = await getLatestVideoSnapshot(sql, aid);
			const now = Date.now();
			const lastSnapshotedAt = latestSnapshot?.time ?? now;
			const interval = 168;
			logger.log(
				`Scheduled archive snapshot for aid ${aid} in ${interval} hours.`,
				"mq",
				"fn:archiveSnapshotsWorker"
			);
			const targetTime = lastSnapshotedAt + interval * HOUR;
			await scheduleSnapshot(sql, aid, "archive", targetTime);
			if (now - startedAt > 250 * MINUTE) {
				return;
			}
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:archiveSnapshotsWorker");
	} finally {
		await lockManager.releaseLock("dispatchArchiveSnapshots");
	}
};
