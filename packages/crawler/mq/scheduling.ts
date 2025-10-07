import { findClosestSnapshot, getLatestSnapshot, hasAtLeast2Snapshots } from "db/snapshotSchedule";
import { truncate } from "utils/truncate";
import { closetMilestone } from "./exec/snapshotTick";
import { HOUR, MINUTE } from "@core/lib";
import type { Psql } from "@core/db/psql.d";
import { updateETA } from "db/eta";

const log = (value: number, base: number = 10) => Math.log(value) / Math.log(base);

const getFactor = (x: number) => {
	const a = 1.054;
	const b = 4.5;
	const c = 100;
	const u = 0.601;
	const g = 455;
	if (x > g) {
		return log(b / log(x + 1), a);
	} else {
		return log(b / log(x + c), a) + u;
	}
};

/*
 * Returns the minimum ETA in hours for the next snapshot
 * @param client - Postgres client
 * @param aid - aid of the video
 * @returns ETA in hours
 */
export const getAdjustedShortTermETA = async (sql: Psql, aid: number) => {
	const latestSnapshot = await getLatestSnapshot(sql, aid);
	// Immediately dispatch a snapshot if there is no snapshot yet
	if (!latestSnapshot) return 0;
	const snapshotsEnough = await hasAtLeast2Snapshots(sql, aid);
	if (!snapshotsEnough) return 0;

	const currentTimestamp = new Date().getTime();
	const timeIntervals = [3 * MINUTE, 20 * MINUTE, HOUR, 3 * HOUR, 6 * HOUR, 72 * HOUR];
	const originalWeight = [3, 5, 3, 2, 2, 3];
	const weight = originalWeight.map((x) => x / originalWeight.reduce((a, b) => a + b, 0));
	const DELTA = 0.00001;
	let minETAHours = Infinity;
	let avgSpeed = 0;

	const target = closetMilestone(latestSnapshot.views);
	const viewsToIncrease = target - latestSnapshot.views;
	const factor = truncate(getFactor(viewsToIncrease), 4.5, 100);

	for (const timeInterval of timeIntervals) {
		const date = new Date(currentTimestamp - timeInterval);
		const snapshot = await findClosestSnapshot(sql, aid, date);
		if (!snapshot) continue;
		const hoursDiff = (latestSnapshot.created_at - snapshot.created_at) / HOUR;
		const viewsDiff = latestSnapshot.views - snapshot.views;
		if (viewsDiff <= 0) continue;
		const speed = viewsDiff / (hoursDiff + DELTA);
		avgSpeed += speed * weight[timeIntervals.indexOf(timeInterval)];
		const eta = viewsToIncrease / (speed + DELTA);
		const adjustedETA = eta / factor;
		if (adjustedETA < minETAHours) {
			minETAHours = adjustedETA;
		}
	}

	if (isNaN(minETAHours)) {
		minETAHours = Infinity;
	}

	const avgETAHours = viewsToIncrease / (avgSpeed + DELTA);

	await updateETA(sql, aid, avgETAHours, avgSpeed, latestSnapshot.views);

	return minETAHours;
};
