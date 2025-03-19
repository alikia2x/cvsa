import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getVideosNearMilestone } from "lib/db/snapshot.ts";
import { findClosestSnapshot, getLatestSnapshot, videoHasActiveSchedule } from "lib/db/snapshotSchedule.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { HOUR, MINUTE } from "$std/datetime/constants.ts";

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		// TODO: implement
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

const getAdjustedShortTermETA = async (client: Client, aid: number) => {
	const latestSnapshot = await getLatestSnapshot(client, aid);
	// Immediately dispatch a snapshot if there is no snapshot yet
	if (!latestSnapshot) return 0;

	const currentTimestamp = Date.now();
	const timeIntervals = [20 * MINUTE, 1 * HOUR, 3 * HOUR, 6 * HOUR];
	const DELTA = 0.00001;
	let minETAHours = Infinity;

	for (const timeInterval of timeIntervals) {
		const date = new Date(currentTimestamp - timeInterval);
		const snapshot = await findClosestSnapshot(client, aid, date);
		if (!snapshot) continue;
		const hoursDiff = (currentTimestamp - snapshot.created_at) / HOUR;
		const viewsDiff = snapshot.views - latestSnapshot.views;
		const speed = viewsDiff / (hoursDiff + DELTA);
		const target = closetMilestone(latestSnapshot.views);
		const viewsToIncrease = target - latestSnapshot.views;
		const eta = viewsToIncrease / (speed + DELTA);
		const factor = log(2.97 / log(viewsToIncrease + 1), 1.14);
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
			if (await videoHasActiveSchedule(client, video.aid)) continue;
			const eta = await getAdjustedShortTermETA(client, video.aid);
			if (eta > 72) continue;
			// TODO: dispatch snapshot job
		}
	} catch (_e) {
		//
	} finally {
		client.release();
	}
};

export const takeSnapshotForVideoWorker = async (_job: Job) => {
	// TODO: implement
};
