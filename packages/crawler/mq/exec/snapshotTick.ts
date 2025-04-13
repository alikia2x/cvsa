import { Job } from "bullmq";
import { db } from "db/init.ts";
import {
	bulkGetVideosWithoutProcessingSchedules,
	bulkSetSnapshotStatus,
	getBulkSnapshotsInNextSecond,
	getSnapshotsInNextSecond,
	scheduleSnapshot,
	setSnapshotStatus,
	snapshotScheduleExists,
	videoHasProcessingSchedule,
} from "db/snapshotSchedule.ts";
import { HOUR, MINUTE, SECOND } from "@std/datetime";
import logger from "log/logger.ts";
import { SnapshotQueue } from "mq/index.ts";
import { insertVideoSnapshot } from "mq/task/getVideoStats.ts";
import { NetSchedulerError } from "@core/net/delegate.ts";
import { getBiliVideoStatus, setBiliVideoStatus } from "db/allData.ts";
import { getSongsPublihsedAt } from "db/songs.ts";
import { getAdjustedShortTermETA } from "../scheduling.ts";
import { getRegularSnapshotInterval } from "../task/regularSnapshotInterval.ts";

const priorityMap: { [key: string]: number } = {
	"milestone": 1,
	"normal": 3,
};

const snapshotTypeToTaskMap: { [key: string]: string } = {
	"milestone": "snapshotMilestoneVideo",
	"normal": "snapshotVideo",
	"new": "snapshotMilestoneVideo",
};

export const bulkSnapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const schedules = await getBulkSnapshotsInNextSecond(client);
		const count = schedules.length;
		const groups = Math.ceil(count / 30);
		for (let i = 0; i < groups; i++) {
			const group = schedules.slice(i * 30, (i + 1) * 30);
			const aids = group.map((schedule) => Number(schedule.aid));
			const filteredAids = await bulkGetVideosWithoutProcessingSchedules(client, aids);
			if (filteredAids.length === 0) continue;
			await bulkSetSnapshotStatus(client, filteredAids, "processing");
			const schedulesData = group.map((schedule) => {
				return {
					aid: Number(schedule.aid),
					id: Number(schedule.id),
					type: schedule.type,
					created_at: schedule.created_at,
					started_at: schedule.started_at,
					finished_at: schedule.finished_at,
					status: schedule.status,
				};
			});
			await SnapshotQueue.add("bulkSnapshotVideo", {
				schedules: schedulesData,
			}, { priority: 3 });
		}
		return `OK`;
	} catch (e) {
		logger.error(e as Error);
	} finally {
		client.release();
	}
};

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const schedules = await getSnapshotsInNextSecond(client);
		for (const schedule of schedules) {
			if (await videoHasProcessingSchedule(client, Number(schedule.aid))) {
				continue;
			}
			let priority = 3;
			if (schedule.type && priorityMap[schedule.type]) {
				priority = priorityMap[schedule.type];
			}
			const aid = Number(schedule.aid);
			await setSnapshotStatus(client, schedule.id, "processing");
			await SnapshotQueue.add("snapshotVideo", {
				aid: Number(aid),
				id: Number(schedule.id),
				type: schedule.type ?? "normal",
			}, { priority });
		}
		return `OK`;
	} catch (e) {
		logger.error(e as Error);
	} finally {
		client.release();
	}
};

export const closetMilestone = (views: number) => {
	if (views < 100000) return 100000;
	if (views < 1000000) return 1000000;
	return 10000000;
};

export const takeSnapshotForVideoWorker = async (job: Job) => {
	const id = job.data.id;
	const aid = Number(job.data.aid);
	const type = job.data.type;
	const task = snapshotTypeToTaskMap[type] ?? "snapshotVideo";
	const client = await db.connect();
	const retryInterval = type === "milestone" ? 5 * SECOND : 2 * MINUTE;
	const exists = await snapshotScheduleExists(client, id);
	if (!exists) {
		client.release();
		return;
	}
	const status = await getBiliVideoStatus(client, aid);
	if (status !== 0) {
		client.release();
		return `REFUSE_WORKING_BILI_STATUS_${status}`;
	}
	try {
		await setSnapshotStatus(client, id, "processing");
		const stat = await insertVideoSnapshot(client, aid, task);
		if (typeof stat === "number") {
			await setBiliVideoStatus(client, aid, stat);
			await setSnapshotStatus(client, id, "completed");
			return `GET_BILI_STATUS_${stat}`;
		}
		await setSnapshotStatus(client, id, "completed");
		if (type === "normal") {
			const interval = await getRegularSnapshotInterval(client, aid);
			logger.log(`Scheduled regular snapshot for aid ${aid} in ${interval} hours.`, "mq");
			await scheduleSnapshot(client, aid, type, Date.now() + interval * HOUR);
			return `DONE`;
		} else if (type === "new") {
			const publihsedAt = await getSongsPublihsedAt(client, aid);
			const timeSincePublished = stat.time - publihsedAt!;
			const viewsPerHour = stat.views / timeSincePublished * HOUR;
			if (timeSincePublished > 48 * HOUR) {
				return `DONE`;
			}
			if (timeSincePublished > 2 * HOUR && viewsPerHour < 10) {
				return `DONE`;
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
		if (type !== "milestone") return `DONE`;
		const eta = await getAdjustedShortTermETA(client, aid);
		if (eta > 144) return "ETA_TOO_LONG";
		const now = Date.now();
		const targetTime = now + eta * HOUR;
		await scheduleSnapshot(client, aid, type, targetTime);
		await setSnapshotStatus(client, id, "completed");
		return `DONE`;
	} catch (e) {
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
	} finally {
		client.release();
	}
};

export const scheduleCleanupWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const query = `
			SELECT id, aid, type 
			FROM snapshot_schedule
			WHERE status IN ('pending', 'processing')
				AND started_at < NOW() - INTERVAL '30 minutes'
		`;
		const { rows } = await client.queryObject<{ id: bigint; aid: bigint; type: string }>(query);
		if (rows.length === 0) return;
		for (const row of rows) {
			const id = Number(row.id);
			const aid = Number(row.aid);
			const type = row.type;
			await setSnapshotStatus(client, id, "timeout");
			await scheduleSnapshot(client, aid, type, Date.now() + 10 * SECOND);
			logger.log(
				`Schedule ${id} has no response received for 5 minutes, rescheduled.`,
				"mq",
				"fn:scheduleCleanupWorker",
			);
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:scheduleCleanupWorker");
	} finally {
		client.release();
	}
};
