import { sql } from "@core/db/dbNew";
import { MINUTE } from "@core/lib";
import logger from "@core/log";
import { lockManager } from "@core/mq/lockManager";
import type { Job } from "bullmq";
import {
	formatDistanceStrict,
	formatDuration,
	intervalToDuration,
	nextMonday,
	nextSaturday,
} from "date-fns";
import {
	getCommonArchiveAids,
	getVideosWithoutActiveSnapshotScheduleByType,
	scheduleSnapshot,
} from "db/snapshotSchedule";

function randomTimestampBetween(start: Date, end: Date) {
	const startMs = start.getTime();
	const endMs = end.getTime();
	const randomMs = startMs + Math.random() * (endMs - startMs);
	return Math.floor(randomMs);
}

const getRandomTimeInNextWeek = (): number => {
	const secondMonday = nextMonday(new Date());
	const thirdMonday = nextMonday(secondMonday);
	return randomTimestampBetween(secondMonday, thirdMonday);
};

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
			const now = Date.now();
			const date = new Date();
			const formatted = formatDistanceStrict(date, nextSaturday(date).getTime(), {
				unit: "hour",
			});
			logger.log(
				`Scheduled archive snapshot for aid ${aid} in ${formatted}.`,
				"mq",
				"fn:archiveSnapshotsWorker"
			);
			await scheduleSnapshot(sql, aid, "archive", nextSaturday(date).getTime());
			if (now - startedAt > 30 * MINUTE) {
				return;
			}
		}
		const aids2 = await getCommonArchiveAids(sql);
		for (const rawAid of aids2) {
			const aid = Number(rawAid);
			const now = Date.now();
			const targetTime = getRandomTimeInNextWeek();
			const interval = intervalToDuration({
				start: new Date(),
				end: new Date(targetTime),
			});
			const formatted = formatDuration(interval, { format: ["days", "hours"] });

			logger.log(
				`Scheduled common archive snapshot for aid ${aid} in ${formatted}.`,
				"mq",
				"fn:archiveSnapshotsWorker"
			);
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
