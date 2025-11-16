import { db, eta as etaTable } from "@core/drizzle";
import { eq } from "drizzle-orm";
import { MINUTE, HOUR, getClosetMilestone } from "@core/lib";
import { getLatestSnapshot, getClosestSnapshot } from "@core/db";

export const getGroundTruthMilestoneETA = async (
	aid: number,
	targetViews?: number
): Promise<number> => {
	const DELTA = 1e-5;
	let minETAHours = Infinity;
	const timeIntervals = [3 * MINUTE, 20 * MINUTE, HOUR, 3 * HOUR, 6 * HOUR, 72 * HOUR];
	const currentTimestamp = new Date().getTime();
	const latestSnapshot = await getLatestSnapshot(aid);
	const latestSnapshotTime = new Date(latestSnapshot.time).getTime();
	for (const timeInterval of timeIntervals) {
		const date = new Date(currentTimestamp - timeInterval);
		const snapshot = await getClosestSnapshot(aid, date);
		if (!snapshot) continue;
		const currentSnapshotTime = new Date(snapshot.created_at).getTime();
		const hoursDiff = (latestSnapshotTime - currentSnapshotTime) / HOUR;
		const viewsDiff = latestSnapshot.views - snapshot.views;
		if (viewsDiff <= 0) continue;
		const speed = viewsDiff / (hoursDiff + DELTA);
		const target = targetViews || getClosetMilestone(latestSnapshot.views);
		const viewsToIncrease = target - latestSnapshot.views;
		const eta = viewsToIncrease / (speed + DELTA);
		if (eta < minETAHours) {
			minETAHours = eta;
		}
	}
	return minETAHours;
};

export const getMilestoneETA = async (aid: number) => {
	const data = await db.select().from(etaTable).where(eq(etaTable.aid, aid)).limit(1);
	if (data.length > 0) {
		return data[0].eta;
	}
	return getGroundTruthMilestoneETA(aid);
};
