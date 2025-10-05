import { Job } from "bullmq";
import { getVideosWithoutActiveSnapshotScheduleByType, scheduleSnapshot } from "db/snapshotSchedule";
import logger from "@core/log";
import { lockManager } from "@core/mq/lockManager";
import { getLatestVideoSnapshot } from "db/snapshot";
import { MINUTE } from "@core/lib";
import { sql } from "@core/db/dbNew";

function getNextSaturdayMidnightTimestamp(): number {
	const now = new Date();
	const currentDay = now.getDay();

	let daysUntilNextSaturday = (6 - currentDay + 7) % 7;

	if (daysUntilNextSaturday === 0) {
		daysUntilNextSaturday = 7;
	}

	const nextSaturday = new Date(now);
	nextSaturday.setDate(nextSaturday.getDate() + daysUntilNextSaturday);
	nextSaturday.setHours(0, 0, 0, 0);

	return nextSaturday.getTime();
}

export const archiveSnapshotsWorker = async (_job: Job) => {
	try {
		const startedAt = Date.now();
		if (await lockManager.isLocked("dispatchArchiveSnapshots")) {
			logger.log("dispatchArchiveSnapshots is already running", "mq");
			return;
		}
		await lockManager.acquireLock("dispatchArchiveSnapshots", 30 * 60);
		const aids = await getVideosWithoutActiveSnapshotScheduleByType(sql, "archive");
		for (const rawAid of aids) {
			const aid = Number(rawAid);
			const latestSnapshot = await getLatestVideoSnapshot(sql, aid);
			const now = Date.now();
			const lastSnapshotedAt = latestSnapshot?.time ?? now;
			const nextSatMidnight = getNextSaturdayMidnightTimestamp();
			const interval = nextSatMidnight - now;
			logger.log(
				`Scheduled archive snapshot for aid ${aid} in ${interval} hours.`,
				"mq",
				"fn:archiveSnapshotsWorker"
			);
			const targetTime = lastSnapshotedAt + interval;
			await scheduleSnapshot(sql, aid, "archive", targetTime);
			if (now - startedAt > 30 * MINUTE) {
				return;
			}
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:archiveSnapshotsWorker");
	} finally {
		await lockManager.releaseLock("dispatchArchiveSnapshots");
	}
};
