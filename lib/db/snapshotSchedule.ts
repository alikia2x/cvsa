import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

/*
    Returns true if the specified `aid` has at least one record with "pending" or "processing" status.
*/
export async function videoHasActiveSchedule(client: Client, aid: number) {
	const res = await client.queryObject<{ status: string }>(
		`SELECT status FROM snapshot_schedule WHERE aid = $1 AND (status = 'pending' OR status = 'processing')`,
		[aid],
	);
	return res.rows.length > 0;
}

interface Snapshot {
	created_at: number;
	views: number;
}

export async function findClosestSnapshot(
	client: Client,
	aid: number,
	targetTime: Date,
): Promise<Snapshot | null> {
	const query = `
        SELECT created_at, views FROM video_snapshot
        WHERE aid = $1
        ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - $2::timestamptz))) ASC
        LIMIT 1
    `;
	const result = await client.queryObject<{ created_at: string; views: number }>(
		query,
		[aid, targetTime.toISOString()],
	);
	if (result.rows.length === 0) return null;
	const row = result.rows[0];
	return {
		created_at: new Date(row.created_at).getTime(),
		views: row.views,
	};
}

export async function getLatestSnapshot(client: Client, aid: number): Promise<Snapshot | null>{
	const res = await client.queryObject<{ created_at: string; views: number }>(
		`SELECT created_at, views FROM video_snapshot WHERE aid = $1 ORDER BY created_at DESC LIMIT 1`,
		[aid],
	);
	if (res.rows.length === 0) return null;
	const row = res.rows[0];
	return {
		created_at: new Date(row.created_at).getTime(),
		views: row.views,
	}
}
