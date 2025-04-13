import { Job } from "npm:bullmq@5.45.2";
import { withDbConnection } from "db/withConnection.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { scheduleSnapshot, setSnapshotStatus, snapshotScheduleExists } from "db/snapshotSchedule.ts";
import logger from "log/logger.ts";
import { HOUR, MINUTE, SECOND } from "@std/datetime";
import { lockManager } from "mq/lockManager.ts";
import { getBiliVideoStatus, setBiliVideoStatus } from "db/allData.ts";
import { insertVideoSnapshot } from "mq/task/getVideoStats.ts";
import { getSongsPublihsedAt } from "db/songs.ts";
import { getAdjustedShortTermETA } from "mq/scheduling.ts";
import { NetSchedulerError } from "@core/net/delegate.ts";

const snapshotTypeToTaskMap: { [key: string]: string } = {
	"milestone": "snapshotMilestoneVideo",
	"normal": "snapshotVideo",
	"new": "snapshotMilestoneVideo",
};

export const snapshotVideoWorker = async (job: Job): Promise<void> => {
	const id = job.data.id;
	const aid = Number(job.data.aid);
	const type = job.data.type;
	const task = snapshotTypeToTaskMap[type] ?? "snapshotVideo";
	const retryInterval = type === "milestone" ? 5 * SECOND : 2 * MINUTE;
	await withDbConnection(async (client: Client) => {
		const exists = await snapshotScheduleExists(client, id);
		if (!exists) {
			return;
		}
		const status = await getBiliVideoStatus(client, aid);
		if (status !== 0) {
			logger.warn(
				`Bilibili return status ${status} when snapshoting for ${aid}.`,
				"mq",
				"fn:dispatchRegularSnapshotsWorker",
			);
			return;
		}
		await setSnapshotStatus(client, id, "processing");
		const stat = await insertVideoSnapshot(client, aid, task);
		if (typeof stat === "number") {
			await setBiliVideoStatus(client, aid, stat);
			await setSnapshotStatus(client, id, "completed");
			logger.warn(
				`Bilibili return status ${status} when snapshoting for ${aid}.`,
				"mq",
				"fn:dispatchRegularSnapshotsWorker",
			);
			return;
		}
		await setSnapshotStatus(client, id, "completed");
		if (type === "new") {
			const publihsedAt = await getSongsPublihsedAt(client, aid);
			const timeSincePublished = stat.time - publihsedAt!;
			const viewsPerHour = stat.views / timeSincePublished * HOUR;
			if (timeSincePublished > 48 * HOUR) {
				return;
			}
			if (timeSincePublished > 2 * HOUR && viewsPerHour < 10) {
				return;
			}
			let intervalMins = 240;
			if (viewsPerHour > 50) {
				intervalMins = 120;
			}
			if (viewsPerHour > 100) {
				intervalMins = 60;
			}
			if (viewsPerHour > 1000) {
				intervalMins = 15;
			}
			await scheduleSnapshot(client, aid, type, Date.now() + intervalMins * MINUTE, true);
		}
		if (type !== "milestone") return;
		const eta = await getAdjustedShortTermETA(client, aid);
		if (eta > 144) {
			const etaHoursString = eta.toFixed(2) + " hrs";
			logger.warn(
				`ETA (${etaHoursString}) too long for milestone snapshot. aid: ${aid}.`,
				"mq",
				"fn:dispatchRegularSnapshotsWorker",
			);
		}
		const now = Date.now();
		const targetTime = now + eta * HOUR;
		await scheduleSnapshot(client, aid, type, targetTime);
		await setSnapshotStatus(client, id, "completed");
		return;
	}, async (e, client) => {
		if (e instanceof NetSchedulerError && e.code === "NO_PROXY_AVAILABLE") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:takeSnapshotForVideoWorker",
			);
			await setSnapshotStatus(client, id, "no_proxy");
			await scheduleSnapshot(client, aid, type, Date.now() + retryInterval);
			return;
		}
		logger.error(e as Error, "mq", "fn:takeSnapshotForVideoWorker");
		await setSnapshotStatus(client, id, "failed");
	}, async () => {
		await lockManager.releaseLock("dispatchRegularSnapshots");
	});
	return;
};
