import { Job } from "npm:bullmq@5.45.2";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getVideosNearMilestone } from "db/snapshot.ts";
import { getAdjustedShortTermETA } from "mq/scheduling.ts";
import { truncate } from "utils/truncate.ts";
import { scheduleSnapshot } from "db/snapshotSchedule.ts";
import logger from "log/logger.ts";
import { HOUR, MINUTE, SECOND } from "@std/datetime";

export const dispatchMilestoneSnapshotsWorker = (_job: Job): Promise<void> =>
	withDbConnection(async (client: Client) => {
		const videos = await getVideosNearMilestone(client);
		for (const video of videos) {
			const aid = Number(video.aid);
			const eta = await getAdjustedShortTermETA(client, aid);
			if (eta > 144) continue;
			const now = Date.now();
			const scheduledNextSnapshotDelay = eta * HOUR;
			const maxInterval = 1 * HOUR;
			const minInterval = 1 * SECOND;
			const delay = truncate(scheduledNextSnapshotDelay, minInterval, maxInterval);
			const targetTime = now + delay;
			await scheduleSnapshot(client, aid, "milestone", targetTime);
			logger.log(`Scheduled milestone snapshot for aid ${aid} in ${(delay / MINUTE).toFixed(2)} mins.`, "mq");
		}
	}, (e) => {
		logger.error(e as Error, "mq", "fn:dispatchMilestoneSnapshotsWorker");
	});
