import { MINUTE, HOUR, getClosetMilestone } from "@core/lib";
import { getLatestSnapshot, getClosestSnapshot } from "@core/db";

export const getMilestoneETA = async (aid: number, targetViews?: number): Promise<number> => {
	const DELTA = 1e-5;
	let minETAHours = Infinity;
	const timeIntervals = [3 * HOUR, 24 * HOUR, 96 * HOUR];
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
