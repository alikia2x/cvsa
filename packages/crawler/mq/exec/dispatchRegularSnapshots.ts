import { sql } from "@core/db/dbNew";
import { HOUR, MINUTE, WEEK } from "@core/lib";
import logger from "@core/log";
import { lockManager } from "@core/mq/lockManager";
import type { Job } from "bullmq";
import { getLatestVideoSnapshot } from "db/snapshot";
import {
	getVideosWithoutActiveSnapshotScheduleByType,
	scheduleSnapshot,
} from "db/snapshotSchedule";
import { getRegularSnapshotInterval } from "mq/task/regularSnapshotInterval";
import { truncate } from "utils/truncate";

export const dispatchRegularSnapshotsWorker = async (_job: Job): Promise<void> => {
	try {
		const startedAt = Date.now();
		if (await lockManager.isLocked("dispatchRegularSnapshots")) {
			logger.log("dispatchRegularSnapshots is already running", "mq");
			return;
		}
		await lockManager.acquireLock("dispatchRegularSnapshots", 30 * 60);

		const aids = await getVideosWithoutActiveSnapshotScheduleByType(sql, "normal");
		for (const rawAid of aids) {
			const aid = Number(rawAid);
			const latestSnapshot = await getLatestVideoSnapshot(sql, aid);
			const now = Date.now();
			const lastSnapshotedAt = latestSnapshot?.time ?? now;
			const interval = await getRegularSnapshotInterval(sql, aid);
			logger.log(`Scheduled regular snapshot for aid ${aid} in ${interval} hours.`, "mq");
			const targetTime = truncate(
				lastSnapshotedAt + interval * HOUR,
				now + 1,
				now + 100000 * WEEK
			);
			await scheduleSnapshot(sql, aid, "normal", targetTime);
			if (now - startedAt > 25 * MINUTE) {
				return;
			}
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:regularSnapshotsWorker");
	} finally {
		await lockManager.releaseLock("dispatchRegularSnapshots");
	}
};
