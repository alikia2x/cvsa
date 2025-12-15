import type { Psql } from "@core/db/psql.d";
import { HOUR } from "@core/lib";
import { findClosestSnapshot, findSnapshotBefore, getLatestSnapshot } from "db/snapshotSchedule";

export const getRegularSnapshotInterval = async (sql: Psql, aid: number) => {
	const now = Date.now();
	const date = new Date(now - 24 * HOUR);
	let oldSnapshot = await findSnapshotBefore(sql, aid, date);
	if (!oldSnapshot) oldSnapshot = await findClosestSnapshot(sql, aid, date);
	const latestSnapshot = await getLatestSnapshot(sql, aid);
	if (!oldSnapshot || !latestSnapshot) return 0;
	if (oldSnapshot.created_at === latestSnapshot.created_at) return 0;
	const hoursDiff = (latestSnapshot.created_at - oldSnapshot.created_at) / HOUR;
	const viewsDiff = latestSnapshot.views - oldSnapshot.views;
	if (viewsDiff === 0) return 72;
	const speedPerDay = (viewsDiff / (hoursDiff + 0.001)) * 24;
	if (speedPerDay < 6) return 36;
	if (speedPerDay < 120) return 24;
	if (speedPerDay < 320) return 12;
	return 6;
};
