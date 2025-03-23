import {DAY, HOUR, MINUTE} from "$std/datetime/constants.ts";
import {Client} from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import {formatTimestampToPsql} from "lib/utils/formatTimestampToPostgre.ts";
import {SnapshotScheduleType} from "./schema.d.ts";
import logger from "lib/log/logger.ts";

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
 * @param allowedCounts The number of snapshots allowed in the 5-minutes windows.
 * @returns The adjusted actual snapshot time
 */
export async function adjustSnapshotTime(
	client: Client,
	expectedStartTime: Date,
	allowedCounts: number = 2000
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
        HAVING COUNT(s.*) < ${allowedCounts}
        ORDER BY w.window_start
        LIMIT 1;
	`;
	const now = new Date();
	const targetTime = expectedStartTime.getTime();
	let start = new Date(targetTime - 2 * HOUR);
	if (start.getTime() <= now.getTime()) {
		start = now;
	}
	const startTruncated = truncateTo5MinInterval(start);
	const end = new Date(startTruncated.getTime() + 1 * DAY);

	const windowResult = await client.queryObject<{ window_start: Date }>(
		findWindowQuery,
		[startTruncated, end],
	);


	const windowStart = windowResult.rows[0]?.window_start;
	if (!windowStart) {
		return expectedStartTime;
	}

	if (windowStart.getTime() > new Date().getTime() + 5 * MINUTE) {
		const randomDelay = Math.floor(Math.random() * 5 * MINUTE);
		return new Date(windowStart.getTime() + randomDelay);
	} else {
		return expectedStartTime;
	}
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
