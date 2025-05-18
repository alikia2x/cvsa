import { Job } from "bullmq";
import { scheduleSnapshot, setSnapshotStatus, snapshotScheduleExists } from "db/snapshotSchedule.ts";
import logger from "@core/log/logger.ts";
import { HOUR, MINUTE, SECOND } from "@core/const/time.ts";
import { lockManager } from "@core/mq/lockManager.ts";
import { getBiliVideoStatus, setBiliVideoStatus } from "../../db/bilibili_metadata.ts";
import { insertVideoSnapshot } from "mq/task/getVideoStats.ts";
import { getSongsPublihsedAt } from "db/songs.ts";
import { getAdjustedShortTermETA } from "mq/scheduling.ts";
import { NetSchedulerError } from "@core/net/delegate.ts";
import { sql } from "@core/db/dbNew.ts";

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
	try {
		const exists = await snapshotScheduleExists(sql, id);
		if (!exists) {
			return;
		}
		const status = await getBiliVideoStatus(sql, aid);
		if (status !== 0) {
			logger.warn(
				`Video ${aid} has status ${status} in the database. Abort snapshoting.`,
				"mq",
				"fn:dispatchRegularSnapshotsWorker",
			);
			return;
		}
		await setSnapshotStatus(sql, id, "processing");
		const stat = await insertVideoSnapshot(sql, aid, task);
		if (typeof stat === "number") {
			await setBiliVideoStatus(sql, aid, stat);
			await setSnapshotStatus(sql, id, "bili_error");
			logger.warn(
				`Bilibili return status ${status} when snapshoting for ${aid}.`,
				"mq",
				"fn:dispatchRegularSnapshotsWorker",
			);
			return;
		}
		await setSnapshotStatus(sql, id, "completed");
		if (type === "new") {
			const publihsedAt = await getSongsPublihsedAt(sql, aid);
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
			await scheduleSnapshot(sql, aid, type, Date.now() + intervalMins * MINUTE, true);
		}
		if (type !== "milestone") return;
		const eta = await getAdjustedShortTermETA(sql, aid);
		if (eta > 144) {
			const etaHoursString = eta.toFixed(2) + " hrs";
			logger.warn(
				`ETA (${etaHoursString}) too long for milestone snapshot. aid: ${aid}.`,
				"mq",
				"fn:snapshotVideoWorker",
			);
			return;
		}
		const now = Date.now();
		const targetTime = now + eta * HOUR;
		await scheduleSnapshot(sql, aid, type, targetTime);
		await setSnapshotStatus(sql, id, "completed");
		return;
	}
	catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_PROXY_AVAILABLE") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:snapshotVideoWorker",
			);
			await setSnapshotStatus(sql, id, "no_proxy");
			await scheduleSnapshot(sql, aid, type, Date.now() + retryInterval);
			return;
		}
		else if (e instanceof NetSchedulerError && e.code === "ALICLOUD_PROXY_ERR") {
			logger.warn(
				`Failed to proxy request for aid ${job.data.aid}: ${e.message}`,
				"mq",
				"fn:snapshotVideoWorker",
			);
			await setSnapshotStatus(sql, id, "failed");
			await scheduleSnapshot(sql, aid, type, Date.now() + retryInterval);
		}
		logger.error(e as Error, "mq", "fn:snapshotVideoWorker");
		await setSnapshotStatus(sql, id, "failed");
	}
};
