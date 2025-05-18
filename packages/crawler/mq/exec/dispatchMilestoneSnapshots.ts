import { Job } from "bullmq";
import { getVideosNearMilestone } from "db/snapshot.ts";
import { getAdjustedShortTermETA } from "mq/scheduling.ts";
import { truncate } from "utils/truncate.ts";
import { scheduleSnapshot } from "db/snapshotSchedule.ts";
import logger from "@core/log/logger.ts";
import { HOUR, MINUTE, SECOND } from "@core/const/time.ts";
import { sql } from "@core/db/dbNew";

export const dispatchMilestoneSnapshotsWorker = async (_job: Job) => {
	try {
		const videos = await getVideosNearMilestone(sql);
		for (const video of videos) {
			const aid = Number(video.aid);
			const eta = await getAdjustedShortTermETA(sql, aid);
			if (eta > 144) continue;
			const now = Date.now();
			const scheduledNextSnapshotDelay = eta * HOUR;
			const maxInterval = 1.2 * HOUR;
			const minInterval = 2 * SECOND;
			const delay = truncate(scheduledNextSnapshotDelay, minInterval, maxInterval);
			const targetTime = now + delay;
			await scheduleSnapshot(sql, aid, "milestone", targetTime);
			logger.log(`Scheduled milestone snapshot for aid ${aid} in ${(delay / MINUTE).toFixed(2)} mins.`, "mq");
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:dispatchMilestoneSnapshotsWorker");
	};
}