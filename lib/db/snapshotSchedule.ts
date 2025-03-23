import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { formatTimestampToPsql } from "lib/utils/formatTimestampToPostgre.ts";
import { SnapshotScheduleType } from "./schema.d.ts";
import logger from "lib/log/logger.ts";
import { MINUTE } from "$std/datetime/constants.ts";

export async function snapshotScheduleExists(client: Client, id: number) {
	const res = await client.queryObject<{ id: number }>(
		`SELECT id FROM snapshot_schedule WHERE id = $1`,
		[id],
	);
	return res.rows.length > 0;
}

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

export async function hasAtLeast2Snapshots(client: Client, aid: number) {
	const res = await client.queryObject<{ count: number }>(
		`SELECT COUNT(*) FROM video_snapshot WHERE aid = $1`,
		[aid],
	);
	return res.rows[0].count >= 2;
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
	if (await videoHasActiveSchedule(client, aid)) return;
	const allowedCount = type === "milestone" ? 2000 : 800;
	const adjustedTime = await adjustSnapshotTime(client, new Date(targetTime), allowedCount);
	logger.log(`Scheduled snapshot for ${aid} at ${adjustedTime.toISOString()}`, "mq", "fn:scheduleSnapshot");
	return client.queryObject(
		`INSERT INTO snapshot_schedule (aid, type, started_at) VALUES ($1, $2, $3)`,
		[aid, type, adjustedTime.toISOString()],
	);
}

/**
 * Adjust the trigger time of the snapshot to ensure it does not exceed the frequency limit
 * @param client PostgreSQL client
 * @param expectedStartTime The expected snapshot time
 * @param allowedCounts The number of snapshots allowed in a 5-minute window (default: 2000)
 * @returns The adjusted actual snapshot time within the first available window
 */
export async function adjustSnapshotTime(
	client: Client,
	expectedStartTime: Date,
	allowedCounts: number = 2000,
): Promise<Date> {
	// Query to find the closest available window by checking both past and future windows
	const findWindowQuery = `
		WITH base AS (
			SELECT 
				date_trunc('minute', $1::timestamp) 
				- (EXTRACT(minute FROM $1::timestamp)::int % 5 * INTERVAL '1 minute') AS base_time
		),
		offsets AS (
			SELECT generate_series(-100, 100) AS "offset"
		),
		candidate_windows AS (
			SELECT
				(base.base_time + ("offset" * INTERVAL '5 minutes')) AS window_start,
				ABS("offset") AS distance
			FROM base
			CROSS JOIN offsets
		)
		SELECT 
			window_start
		FROM 
			candidate_windows cw
		LEFT JOIN 
			snapshot_schedule s 
		ON 
			s.started_at >= cw.window_start 
			AND s.started_at < cw.window_start + INTERVAL '5 minutes'
			AND s.status = 'pending'
		GROUP BY 
			cw.window_start, cw.distance
		HAVING 
			COUNT(s.*) < $2
		ORDER BY 
			cw.distance, cw.window_start
		LIMIT 1;
  	`;

	try {
		// Execute query to find the first available window
		const windowResult = await client.queryObject<{ window_start: Date }>(
			findWindowQuery,
			[expectedStartTime, allowedCounts],
		);

		// If no available window found, return original time (may exceed limit)
		if (windowResult.rows.length === 0) {
			return expectedStartTime;
		}

		// Get the target window start time
		const windowStart = windowResult.rows[0].window_start;

		// Add random delay within the 5-minute window to distribute load
		const randomDelay = Math.floor(Math.random() * 5 * MINUTE);
		return new Date(windowStart.getTime() + randomDelay);
	} catch {
		return expectedStartTime; // Fallback to original time
	}
}

export async function getSnapshotsInNextSecond(client: Client) {
	const query = `
		SELECT *
		FROM snapshot_schedule
		WHERE started_at <= NOW() + INTERVAL '1 seconds' AND status = 'pending'
		ORDER BY
			CASE
				WHEN type = 'milestone' THEN 0
				ELSE 1
			END,
			started_at
		LIMIT 3;
	`;
	const res = await client.queryObject<SnapshotScheduleType>(query, []);
	return res.rows;
}

export async function setSnapshotStatus(client: Client, id: number, status: string) {
	return await client.queryObject(
		`UPDATE snapshot_schedule SET status = $2 WHERE id = $1`,
		[id, status],
	);
}

export async function getVideosWithoutActiveSnapshotSchedule(client: Client) {
	const query: string = `
		SELECT s.aid
		FROM songs s
		LEFT JOIN snapshot_schedule ss ON s.aid = ss.aid AND (ss.status = 'pending' OR ss.status = 'processing')
		WHERE ss.aid IS NULL
	`;
	const res = await client.queryObject<{ aid: number }>(query, []);
	return res.rows.map((r) => Number(r.aid));
}
