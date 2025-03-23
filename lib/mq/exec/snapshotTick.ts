import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getVideosNearMilestone } from "lib/db/snapshot.ts";
import {
	findClosestSnapshot,
	getLatestSnapshot,
	getSnapshotsInNextSecond,
	hasAtLeast2Snapshots,
	scheduleSnapshot,
	setSnapshotStatus,
	videoHasActiveSchedule,
	videoHasProcessingSchedule,
} from "lib/db/snapshotSchedule.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { HOUR, MINUTE, SECOND } from "$std/datetime/constants.ts";
import logger from "lib/log/logger.ts";
import { SnapshotQueue } from "lib/mq/index.ts";
import { insertVideoSnapshot } from "../task/getVideoStats.ts";
import { NetSchedulerError } from "../scheduler.ts";
import { setBiliVideoStatus } from "../../db/allData.ts";
import {truncate} from "../../utils/truncate.ts";

const priorityMap: { [key: string]: number } = {
	"milestone": 1,
};

const snapshotTypeToTaskMap: { [key: string]: string } = {
	"milestone": "snapshotMilestoneVideo",
	"normal": "snapshotVideo",
};

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const schedules = await getSnapshotsInNextSecond(client);
		for (const schedule of schedules) {
			let priority = 3;
			if (schedule.type && priorityMap[schedule.type]) {
				priority = priorityMap[schedule.type];
			}
			const aid = Number(schedule.aid);
			await SnapshotQueue.add("snapshotVideo", {
				aid: aid,
				id: schedule.id,
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

	const currentTimestamp = new Date().getTime()
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
		factor = truncate(factor, 3, 100)
		const adjustedETA = eta / factor;
		if (adjustedETA < minETAHours) {
			minETAHours = adjustedETA;
		}
	}
	return minETAHours;
};

export const collectMilestoneSnapshotsWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const videos = await getVideosNearMilestone(client);
		for (const video of videos) {
			const aid = Number(video.aid)
			if (await videoHasActiveSchedule(client, aid)) continue;
			const eta = await getAdjustedShortTermETA(client, aid);
			if (eta > 72) continue;
			const now = Date.now();
			const scheduledNextSnapshotDelay = eta * HOUR;
			const maxInterval = 60 * MINUTE;
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

export const takeSnapshotForVideoWorker = async (job: Job) => {
	const id = job.data.id;
	const aid = job.data.aid;
	const task = snapshotTypeToTaskMap[job.data.type] ?? "snapshotVideo";
	const client = await db.connect();
	try {
		if (await videoHasProcessingSchedule(client, aid)) {
			return `ALREADY_PROCESSING`;
		}
		await setSnapshotStatus(client, id, "processing");
		const stat = await insertVideoSnapshot(client, aid, task);
		if (typeof stat === "number") {
			await setBiliVideoStatus(client, aid, stat);
			await setSnapshotStatus(client, id, "completed");
			return `BILI_STATUS_${stat}`;
		}
		const eta = await getAdjustedShortTermETA(client, aid);
		if (eta > 72) return "ETA_TOO_LONG";
		const now = Date.now();
		const targetTime = now + eta * HOUR;
		await setSnapshotStatus(client, id, "completed");
		await scheduleSnapshot(client, aid, "milestone", targetTime);
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_PROXY_AVAILABLE") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:takeSnapshotForVideoWorker",
			);
			await setSnapshotStatus(client, id, "completed");
			await scheduleSnapshot(client, aid, "milestone", Date.now() + 5 * SECOND);
			return;
		}
		logger.error(e as Error, "mq", "fn:takeSnapshotForVideoWorker");
		await setSnapshotStatus(client, id, "failed");
	} finally {
		client.release();
	}
};
