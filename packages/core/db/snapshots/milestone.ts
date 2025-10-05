import { MINUTE, HOUR, getClosetMilestone, getMileStoneETAfactor, truncate } from "@core/lib";
import { getLatestSnapshot, getClosestSnapshot } from "@core/db";

export const getShortTermETA = async (aid: number) => {
	const DELTA = 1e-5;
	let minETAHours = Infinity;
	const timeIntervals = [20 * MINUTE, HOUR, 3 * HOUR, 6 * HOUR, 24 * HOUR, 72 * HOUR, 168 * HOUR];
	const currentTimestamp = new Date().getTime();
	const latestSnapshot = await getLatestSnapshot(aid);
	for (const timeInterval of timeIntervals) {
		const date = new Date(currentTimestamp - timeInterval);
		const snapshot = await getClosestSnapshot(aid, date);
		if (!snapshot) continue;
		const latestSnapshotTime = new Date(latestSnapshot.time).getTime();
		const currentSnapshotTime = new Date(snapshot.created_at).getTime();
		const hoursDiff = (latestSnapshotTime - currentSnapshotTime) / HOUR;
		const viewsDiff = latestSnapshot.views - snapshot.views;
		if (viewsDiff <= 0) continue;
		const speed = viewsDiff / (hoursDiff + DELTA);
		const target = getClosetMilestone(latestSnapshot.views);
		const viewsToIncrease = target - latestSnapshot.views;
		const eta = viewsToIncrease / (speed + DELTA);
		let factor = getMileStoneETAfactor(viewsToIncrease);
		factor = truncate(factor, 4.5, 100);
		const adjustedETA = eta / factor;
		if (adjustedETA < minETAHours) {
			minETAHours = adjustedETA;
		}
	}
};
