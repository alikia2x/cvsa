import { DAY, MINUTE } from "$std/datetime/constants.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { formatTimestampToPsql } from "lib/utils/formatTimestampToPostgre.ts";
import { SnapshotScheduleType } from "./schema.d.ts";

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

export async function videoHasProcessingSchedule(client: Client, aid: number) {
	const res = await client.queryObject<{ status: string }>(
		`SELECT status FROM snapshot_schedule WHERE aid = $1 AND status = 'processing'`,
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
        SELECT created_at, views
        FROM video_snapshot
        WHERE aid = $1
        ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - $2::timestamptz)))
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

export async function getLatestSnapshot(client: Client, aid: number): Promise<Snapshot | null> {
	const res = await client.queryObject<{ created_at: string; views: number }>(
		`SELECT created_at, views FROM video_snapshot WHERE aid = $1 ORDER BY created_at DESC LIMIT 1`,
		[aid],
	);
	if (res.rows.length === 0) return null;
	const row = res.rows[0];
	return {
		created_at: new Date(row.created_at).getTime(),
		views: row.views,
	};
}

/*
 * Returns the number of snapshot schedules within the specified range.
 * @param client The database client.
 * @param start The start time of the range. (Timestamp in milliseconds)
 * @param end The end time of the range. (Timestamp in milliseconds)
 */
export async function getSnapshotScheduleCountWithinRange(client: Client, start: number, end: number) {
	const startTimeString = formatTimestampToPsql(start);
	const endTimeString = formatTimestampToPsql(end);
	const query = `
		SELECT COUNT(*) FROM snapshot_schedule
		WHERE started_at BETWEEN $1 AND $2
		AND status = 'pending'
	`;
	const res = await client.queryObject<{ count: number }>(query, [startTimeString, endTimeString]);
	return res.rows[0].count;
}

/*
 * Creates a new snapshot schedule record.
 * @param client The database client.
 * @param aid The aid of the video.
 * @param targetTime Scheduled time for snapshot. (Timestamp in milliseconds)
 */
export async function scheduleSnapshot(client: Client, aid: number, type: string, targetTime: number) {
	const ajustedTime = await adjustSnapshotTime(client, new Date(targetTime));
	return client.queryObject(
		`INSERT INTO snapshot_schedule (aid, type, started_at) VALUES ($1, $2, $3)`,
		[aid, type, ajustedTime.toISOString()],
	);
}

/**
 * Adjust the trigger time of the snapshot to ensure it does not exceed the frequency limit
 * @param client PostgreSQL client
 * @param expectedStartTime The expected snapshot time
 * @returns The adjusted actual snapshot time
 */
export async function adjustSnapshotTime(
	client: Client,
	expectedStartTime: Date,
): Promise<Date> {
	const findWindowQuery = `
        WITH windows AS (
            SELECT generate_series(
				   $1::timestamp,  -- Start time: current time truncated to the nearest 5-minute window
				   $2::timestamp,  -- End time: 24 hours after the target time window starts
				   INTERVAL '5 MINUTES'
		   ) AS window_start
        )
        SELECT w.window_start
        FROM windows w
        	LEFT JOIN snapshot_schedule s ON s.started_at >= w.window_start
			AND s.started_at < w.window_start + INTERVAL '5 MINUTES'
			AND s.status = 'pending'
        GROUP BY w.window_start
        HAVING COUNT(s.*) < 2000
        ORDER BY w.window_start
        LIMIT 1;
	`;
	for (let i = 0; i < 7; i++) {
		const now = new Date(new Date().getTime() + 5 * MINUTE);
		const nowTruncated = truncateTo5MinInterval(now);
		const currentWindowStart = truncateTo5MinInterval(expectedStartTime);
		const end = new Date(currentWindowStart.getTime() + 1 * DAY);

		const windowResult = await client.queryObject<{ window_start: Date }>(
			findWindowQuery,
			[nowTruncated, end],
		);

		const windowStart = windowResult.rows[0]?.window_start;
		if (!windowStart) {
			continue;
		}

		return windowStart;
	}
	return expectedStartTime;
}

/**
 * Truncate the timestamp to the nearest 5-minute interval
 * @param timestamp The timestamp
 * @returns The truncated time
 */
function truncateTo5MinInterval(timestamp: Date): Date {
	const minutes = timestamp.getMinutes() - (timestamp.getMinutes() % 5);
	return new Date(
		timestamp.getFullYear(),
		timestamp.getMonth(),
		timestamp.getDate(),
		timestamp.getHours(),
		minutes,
		0,
		0,
	);
}

export async function getSnapshotsInNextSecond(client: Client) {
	const query = `
		SELECT * 
		FROM snapshot_schedule 
		WHERE started_at 
		    BETWEEN NOW() - INTERVAL '5 seconds'
			AND NOW() + INTERVAL '1 seconds'
	`;
	const res = await client.queryObject<SnapshotScheduleType>(query, []);
	return res.rows;
}

export async function setSnapshotStatus(client: Client, id: number, status: string) {
	return client.queryObject(
		`UPDATE snapshot_schedule SET status = $2 WHERE id = $1`,
		[id, status],
	);
}
