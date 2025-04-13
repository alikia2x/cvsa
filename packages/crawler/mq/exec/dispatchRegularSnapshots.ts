import { Job } from "npm:bullmq@5.45.2";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getLatestVideoSnapshot } from "db/snapshot.ts";
import { truncate } from "utils/truncate.ts";
import { getVideosWithoutActiveSnapshotSchedule, scheduleSnapshot } from "db/snapshotSchedule.ts";
import logger from "log/logger.ts";
import { HOUR, MINUTE, WEEK } from "@std/datetime";
import { lockManager } from "../lockManager.ts";
import { getRegularSnapshotInterval } from "../task/regularSnapshotInterval.ts";

export const dispatchRegularSnapshotsWorker = (_job: Job): Promise<void> =>
	withDbConnection(async (client: Client) => {
		const startedAt = Date.now();
		if (await lockManager.isLocked("dispatchRegularSnapshots")) {
			logger.log("dispatchRegularSnapshots is already running", "mq");
			return;
		}
		await lockManager.acquireLock("dispatchRegularSnapshots", 30 * 60);

		const aids = await getVideosWithoutActiveSnapshotSchedule(client);
		for (const rawAid of aids) {
			const aid = Number(rawAid);
			const latestSnapshot = await getLatestVideoSnapshot(client, aid);
			const now = Date.now();
			const lastSnapshotedAt = latestSnapshot?.time ?? now;
			const interval = await getRegularSnapshotInterval(client, aid);
			logger.log(`Scheduled regular snapshot for aid ${aid} in ${interval} hours.`, "mq");
			const targetTime = truncate(lastSnapshotedAt + interval * HOUR, now + 1, now + 100000 * WEEK);
			await scheduleSnapshot(client, aid, "normal", targetTime);
			if (now - startedAt > 25 * MINUTE) {
				return;
			}
		}
	}, (e) => {
		logger.error(e as Error, "mq", "fn:regularSnapshotsWorker");
	}, async () => {
		await lockManager.releaseLock("dispatchRegularSnapshots");
	});
