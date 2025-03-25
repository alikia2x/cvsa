import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getLatestVideoSnapshot, getVideosNearMilestone } from "lib/db/snapshot.ts";
import {
	findClosestSnapshot,
	getLatestSnapshot,
	getSnapshotsInNextSecond,
	getVideosWithoutActiveSnapshotSchedule,
	hasAtLeast2Snapshots,
	scheduleSnapshot,
	setSnapshotStatus,
	snapshotScheduleExists,
	videoHasProcessingSchedule,
} from "lib/db/snapshotSchedule.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { HOUR, MINUTE, SECOND, WEEK } from "$std/datetime/constants.ts";
import logger from "lib/log/logger.ts";
import { SnapshotQueue } from "lib/mq/index.ts";
import { insertVideoSnapshot } from "lib/mq/task/getVideoStats.ts";
import { NetSchedulerError } from "lib/mq/scheduler.ts";
import { setBiliVideoStatus } from "lib/db/allData.ts";
import { truncate } from "lib/utils/truncate.ts";
import { lockManager } from "lib/mq/lockManager.ts";
import { getSongsPublihsedAt } from "lib/db/songs.ts";

const priorityMap: { [key: string]: number } = {
	"milestone": 1,
	"normal": 3,
};

const snapshotTypeToTaskMap: { [key: string]: string } = {
	"milestone": "snapshotMilestoneVideo",
	"normal": "snapshotVideo",
	"new": "snapshotMilestoneVideo",
};

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const schedules = await getSnapshotsInNextSecond(client);
		for (const schedule of schedules) {
			if (await videoHasProcessingSchedule(client, schedule.aid)) {
				return `ALREADY_PROCESSING`;
			}
			let priority = 3;
			if (schedule.type && priorityMap[schedule.type]) {
				priority = priorityMap[schedule.type];
			}
			const aid = Number(schedule.aid);
			await setSnapshotStatus(client, schedule.id, "processing");
			await SnapshotQueue.add("snapshotVideo", {
				aid: aid,
				id: Number(schedule.id),
				type: schedule.type ?? "normal",
			}, { priority });
		}
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

const log = (value: number, base: number = 10) => Math.log(value) / Math.log(base);

/*
 * Returns the minimum ETA in hours for the next snapshot
 * @param client - Postgres client
 * @param aid - aid of the video
 * @returns ETA in hours
 */
const getAdjustedShortTermETA = async (client: Client, aid: number) => {
	const latestSnapshot = await getLatestSnapshot(client, aid);
	// Immediately dispatch a snapshot if there is no snapshot yet
	if (!latestSnapshot) return 0;
	const snapshotsEnough = await hasAtLeast2Snapshots(client, aid);
	if (!snapshotsEnough) return 0;

	const currentTimestamp = new Date().getTime();
	const timeIntervals = [3 * MINUTE, 20 * MINUTE, 1 * HOUR, 3 * HOUR, 6 * HOUR];
	const DELTA = 0.00001;
	let minETAHours = Infinity;

	for (const timeInterval of timeIntervals) {
		const date = new Date(currentTimestamp - timeInterval);
		const snapshot = await findClosestSnapshot(client, aid, date);
		if (!snapshot) continue;
		const hoursDiff = (latestSnapshot.created_at - snapshot.created_at) / HOUR;
		const viewsDiff = latestSnapshot.views - snapshot.views;
		if (viewsDiff <= 0) continue;
		const speed = viewsDiff / (hoursDiff + DELTA);
		const target = closetMilestone(latestSnapshot.views);
		const viewsToIncrease = target - latestSnapshot.views;
		const eta = viewsToIncrease / (speed + DELTA);
		let factor = log(2.97 / log(viewsToIncrease + 1), 1.14);
		factor = truncate(factor, 3, 100);
		const adjustedETA = eta / factor;
		if (adjustedETA < minETAHours) {
			minETAHours = adjustedETA;
		}
	}

	if (isNaN(minETAHours)) {
		minETAHours = Infinity;
	}

	return minETAHours;
};

export const collectMilestoneSnapshotsWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const videos = await getVideosNearMilestone(client);
		for (const video of videos) {
			const aid = Number(video.aid);
			const eta = await getAdjustedShortTermETA(client, aid);
			if (eta > 72) continue;
			const now = Date.now();
			const scheduledNextSnapshotDelay = eta * HOUR;
			const maxInterval = 4 * HOUR;
			const minInterval = 1 * SECOND;
			const delay = truncate(scheduledNextSnapshotDelay, minInterval, maxInterval);
			const targetTime = now + delay;
			await scheduleSnapshot(client, aid, "milestone", targetTime);
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:collectMilestoneSnapshotsWorker");
	} finally {
		client.release();
	}
};

const getRegularSnapshotInterval = async (client: Client, aid: number) => {
	const now = Date.now();
	const date = new Date(now - 24 * HOUR);
	const oldSnapshot = await findClosestSnapshot(client, aid, date);
	const latestSnapshot = await getLatestSnapshot(client, aid);
	if (!oldSnapshot || !latestSnapshot) return 0;
	if (oldSnapshot.created_at === latestSnapshot.created_at) return 0;
	const hoursDiff = (latestSnapshot.created_at - oldSnapshot.created_at) / HOUR;
	if (hoursDiff < 8) return 24;
	const viewsDiff = latestSnapshot.views - oldSnapshot.views;
	if (viewsDiff === 0) return 72;
	const speedPerDay = viewsDiff / hoursDiff * 24;
	if (speedPerDay < 6) return 36;
	if (speedPerDay < 120) return 24;
	if (speedPerDay < 320) return 12;
	return 6;
};

export const regularSnapshotsWorker = async (_job: Job) => {
	const client = await db.connect();
	const startedAt = Date.now();
	if (await lockManager.isLocked("dispatchRegularSnapshots")) {
		logger.log("dispatchRegularSnapshots is already running", "mq");
		return;
	}
	await lockManager.acquireLock("dispatchRegularSnapshots", 30 * 60);
	try {
		const aids = await getVideosWithoutActiveSnapshotSchedule(client);
		for (const rawAid of aids) {
			const aid = Number(rawAid);
			const latestSnapshot = await getLatestVideoSnapshot(client, aid);
			const now = Date.now();
			const lastSnapshotedAt = latestSnapshot?.time ?? now;
			const interval = await getRegularSnapshotInterval(client, aid);
			const targetTime = truncate(lastSnapshotedAt + interval * HOUR, now + 1, now + 100000 * WEEK);
			await scheduleSnapshot(client, aid, "normal", targetTime);
			if (now - startedAt > 25 * MINUTE) {
				return;
			}
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:regularSnapshotsWorker");
	} finally {
		lockManager.releaseLock("dispatchRegularSnapshots");
		client.release();
	}
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
		return;
	}
	try {
		await setSnapshotStatus(client, id, "processing");
		const stat = await insertVideoSnapshot(client, aid, task);
		if (typeof stat === "number") {
			await setBiliVideoStatus(client, aid, stat);
			await setSnapshotStatus(client, id, "completed");
			return `BILI_STATUS_${stat}`;
		}
		await setSnapshotStatus(client, id, "completed");
		if (type === "normal") {
			const interval = await getRegularSnapshotInterval(client, aid);
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
		if (eta > 72) return "ETA_TOO_LONG";
		const now = Date.now();
		const targetTime = now + eta * HOUR;
		await scheduleSnapshot(client, aid, type, targetTime);
		return `DONE`;
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_PROXY_AVAILABLE") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:takeSnapshotForVideoWorker",
			);
			await setSnapshotStatus(client, id, "completed");
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
				AND started_at < NOW() - INTERVAL '5 minutes'
		`;
		const { rows } = await client.queryObject<{ id: bigint; aid: bigint; type: string }>(query);
		if (rows.length === 0) return;
		for (const row of rows) {
			const id = Number(row.id);
			const aid = Number(row.aid);
			const type = row.type;
			await setSnapshotStatus(client, id, "timeout");
			await scheduleSnapshot(client, aid, type, Date.now() + 10 * SECOND);
			logger.log(`Schedule ${id} has no response received for 5 minutes, rescheduled.`, "mq", "fn:scheduleCleanupWorker")
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:scheduleCleanupWorker");
	} finally {
		client.release();
	}
};
