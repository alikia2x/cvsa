import { findClosestSnapshot, findSnapshotBefore, getLatestSnapshot } from "db/snapshotSchedule.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { HOUR } from "@std/datetime";

export const getRegularSnapshotInterval = async (client: Client, aid: number) => {
	const now = Date.now();
	const date = new Date(now - 24 * HOUR);
	let oldSnapshot = await findSnapshotBefore(client, aid, date);
	if (!oldSnapshot) oldSnapshot = await findClosestSnapshot(client, aid, date);
	const latestSnapshot = await getLatestSnapshot(client, aid);
	if (!oldSnapshot || !latestSnapshot) return 0;
	if (oldSnapshot.created_at === latestSnapshot.created_at) return 0;
	const hoursDiff = (latestSnapshot.created_at - oldSnapshot.created_at) / HOUR;
	if (hoursDiff < 8) return 24;
	const viewsDiff = latestSnapshot.views - oldSnapshot.views;
	if (viewsDiff === 0) return 72;
	const speedPerDay = viewsDiff / (hoursDiff + 0.001) * 24;
	if (speedPerDay < 6) return 36;
	if (speedPerDay < 120) return 24;
	if (speedPerDay < 320) return 12;
	return 6;
};
