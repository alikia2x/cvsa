import { Job } from "bullmq";
import { getVideosNearMilestone } from "db/snapshot";
import { getAdjustedShortTermETA } from "mq/scheduling";
import { truncate } from "utils/truncate";
import { scheduleSnapshot } from "db/snapshotSchedule";
import logger from "@core/log";
import { HOUR, MINUTE, SECOND } from "@core/lib";
import { sql } from "@core/db/dbNew";
import { jobCounter, jobDuration, jobDurationRaw } from "metrics";

export const dispatchMilestoneSnapshotsWorker = async (_job: Job) => {
	const start = Date.now();
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
			logger.log(
				`Scheduled milestone snapshot for aid ${aid} in ${(delay / MINUTE).toFixed(2)} mins.`,
				"mq"
			);
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:dispatchMilestoneSnapshotsWorker");
	} finally {
		const duration = Date.now() - start;

		jobCounter.add(1, { jobName: "dispatchMilestoneSnapshots" });
		jobDurationRaw.record(duration, { jobName: "dispatchMilestoneSnapshots" });
	}
};
