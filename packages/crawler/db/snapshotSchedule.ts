import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { SnapshotScheduleType } from "@core/db/schema";
import logger from "log/logger.ts";
import { MINUTE } from "$std/datetime/constants.ts";
import { redis } from "@core/db/redis.ts";
import { Redis } from "ioredis";
import {parseTimestampFromPsql} from "../utils/formatTimestampToPostgre.ts";

const REDIS_KEY = "cvsa:snapshot_window_counts";

function getCurrentWindowIndex(): number {
	const now = new Date();
	const minutesSinceMidnight = now.getHours() * 60 + now.getMinutes();
	return Math.floor(minutesSinceMidnight / 5);
}

export async function refreshSnapshotWindowCounts(client: Client, redisClient: Redis) {
	const now = new Date();
	const startTime = now.getTime();

	const result = await client.queryObject<{ window_start: Date; count: number }>`
		SELECT 
		date_trunc('hour', started_at) + 
		(EXTRACT(minute FROM started_at)::int / 5 * INTERVAL '5 minutes') AS window_start,
		COUNT(*) AS count
		FROM snapshot_schedule
		WHERE started_at >= NOW() AND status = 'pending' AND started_at <= NOW() + INTERVAL '10 days'
		GROUP BY 1
		ORDER BY window_start
  	`;

	await redisClient.del(REDIS_KEY);

	const currentWindow = getCurrentWindowIndex();

	for (const row of result.rows) {
		const targetOffset = Math.floor((row.window_start.getTime() - startTime) / (5 * MINUTE));
		const offset = currentWindow + targetOffset;
		if (offset >= 0) {
			await redisClient.hset(REDIS_KEY, offset.toString(), Number(row.count));
		}
	}
}

export async function initSnapshotWindowCounts(client: Client, redisClient: Redis) {
	await refreshSnapshotWindowCounts(client, redisClient);
	setInterval(async () => {
		await refreshSnapshotWindowCounts(client, redisClient);
	}, 5 * MINUTE);
}

async function getWindowCount(redisClient: Redis, offset: number): Promise<number> {
	const count = await redisClient.hget(REDIS_KEY, offset.toString());
	return count ? parseInt(count, 10) : 0;
}

export async function snapshotScheduleExists(client: Client, id: number) {
	const res = await client.queryObject<{ id: number }>(
		`SELECT id FROM snapshot_schedule WHERE id = $1`,
		[id],
	);
	return res.rows.length > 0;
}

export async function videoHasActiveSchedule(client: Client, aid: number) {
	const res = await client.queryObject<{ status: string }>(
		`SELECT status FROM snapshot_schedule WHERE aid = $1 AND (status = 'pending' OR status = 'processing')`,
		[aid],
	);
	return res.rows.length > 0;
}

export async function videoHasActiveScheduleWithType(client: Client, aid: number, type: string) {
	const res = await client.queryObject<{ status: string }>(
		`SELECT status FROM snapshot_schedule WHERE aid = $1 AND (status = 'pending' OR status = 'processing') AND type = $2`,
		[aid, type],
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

export async function bulkGetVideosWithoutProcessingSchedules(client: Client, aids: number[]) {
	const res = await client.queryObject<{ aid: number }>(
		`SELECT aid FROM snapshot_schedule WHERE aid = ANY($1) AND status != 'processing' GROUP BY aid`,
		[aids],
	);
	return res.rows.map((row) => row.aid);
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

export async function findSnapshotBefore(
	client: Client,
	aid: number,
	targetTime: Date,
): Promise<Snapshot | null> {
	const query = `
        SELECT created_at, views
        FROM video_snapshot
        WHERE aid = $1
		AND created_at <= $2::timestamptz
        ORDER BY created_at DESC
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

export async function getLatestActiveScheduleWithType(client: Client, aid: number, type: string) {
	const query: string = `
		SELECT *
		FROM snapshot_schedule
		WHERE aid = $1
		  AND type = $2 
		  AND (status = 'pending' OR status = 'processing')
		ORDER BY started_at DESC
		LIMIT 1
	`
	const res = await client.queryObject<SnapshotScheduleType>(query, [aid, type]);
	return res.rows[0];
}

/*
 * Creates a new snapshot schedule record.
 * @param client The database client.
 * @param aid The aid of the video.
 * @param targetTime Scheduled time for snapshot. (Timestamp in milliseconds)
 */
export async function scheduleSnapshot(
	client: Client,
	aid: number,
	type: string,
	targetTime: number,
	force: boolean = false,
) {
	let adjustedTime = new Date(targetTime);
	const hashActiveSchedule = await videoHasActiveScheduleWithType(client, aid, type);
	if (type == "milestone" && hashActiveSchedule) {
		const latestActiveSchedule = await getLatestActiveScheduleWithType(client, aid, type);
		const latestScheduleStartedAt = new Date(parseTimestampFromPsql(latestActiveSchedule.started_at!));
		if (latestScheduleStartedAt > adjustedTime) {
			await client.queryObject(`
                UPDATE snapshot_schedule
                SET started_at = $1
                WHERE id = $2
			`, [adjustedTime, latestActiveSchedule.id]);
			logger.log(
				`Updated snapshot schedule for ${aid} at ${adjustedTime.toISOString()}`,
				"mq",
				"fn:scheduleSnapshot",
			);
			return;
		}
	}
	if (hashActiveSchedule && !force) return;
	if (type !== "milestone" && type !== "new") {
		adjustedTime = await adjustSnapshotTime(new Date(targetTime), 2000, redis);
	}
	logger.log(`Scheduled snapshot for ${aid} at ${adjustedTime.toISOString()}`, "mq", "fn:scheduleSnapshot");
	return client.queryObject(
		`INSERT INTO snapshot_schedule (aid, type, started_at) VALUES ($1, $2, $3)`,
		[aid, type, adjustedTime.toISOString()],
	);
}

export async function bulkScheduleSnapshot(
	client: Client,
	aids: number[],
	type: string,
	targetTime: number,
	force: boolean = false,
) {
	for (const aid of aids) {
		await scheduleSnapshot(client, aid, type, targetTime, force);
	}
}

export async function adjustSnapshotTime(
	expectedStartTime: Date,
	allowedCounts: number = 1000,
	redisClient: Redis,
): Promise<Date> {
	const currentWindow = getCurrentWindowIndex();
	const targetOffset = Math.floor((expectedStartTime.getTime() - Date.now()) / (5 * MINUTE)) - 6;

	const initialOffset = currentWindow + Math.max(targetOffset, 0);

	let timePerIteration: number;
	const MAX_ITERATIONS = 2880;
	let iters = 0;
	const t = performance.now();
	for (let i = initialOffset; i < MAX_ITERATIONS; i++) {
		iters++;
		const offset = i;
		const count = await getWindowCount(redisClient, offset);

		if (count < allowedCounts) {
			await redisClient.hincrby(REDIS_KEY, offset.toString(), 1);

			const startPoint = new Date();
			startPoint.setHours(0, 0, 0, 0);
			const startTime = startPoint.getTime();
			const windowStart = startTime + offset * 5 * MINUTE;
			const randomDelay = Math.floor(Math.random() * 5 * MINUTE);
			const delayedDate = new Date(windowStart + randomDelay);
			const now = new Date();

			if (delayedDate.getTime() < now.getTime()) {
				const elapsed = performance.now() - t;
				timePerIteration = elapsed / (i + 1);
				logger.log(`${timePerIteration.toFixed(3)}ms * ${iters} iterations`, "perf", "fn:adjustSnapshotTime");
				return now;
			}
			const elapsed = performance.now() - t;
			timePerIteration = elapsed / (i + 1);
			logger.log(`${timePerIteration.toFixed(3)}ms * ${iters} iterations`, "perf", "fn:adjustSnapshotTime");
			return delayedDate;
		}
	}
	const elapsed = performance.now() - t;
	timePerIteration = elapsed / MAX_ITERATIONS;
	logger.log(`${timePerIteration.toFixed(3)}ms * ${MAX_ITERATIONS} iterations`, "perf", "fn:adjustSnapshotTime");
	return expectedStartTime;
}

export async function getSnapshotsInNextSecond(client: Client) {
	const query = `
		SELECT *
		FROM snapshot_schedule
		WHERE started_at <= NOW() + INTERVAL '1 seconds' AND status = 'pending' AND type != 'normal'
		ORDER BY
			CASE
				WHEN type = 'milestone' THEN 0
				ELSE 1
			END,
			started_at
		LIMIT 10;
	`;
	const res = await client.queryObject<SnapshotScheduleType>(query, []);
	return res.rows;
}

export async function getBulkSnapshotsInNextSecond(client: Client) {
	const query = `
        SELECT *
        FROM snapshot_schedule
        WHERE (started_at <= NOW() + INTERVAL '15 seconds')
          AND status = 'pending'
          AND (type = 'normal' OR type = 'archive')
        ORDER BY CASE
                     WHEN type = 'normal' THEN 1
                     WHEN type = 'archive' THEN 2
                     END,
                 started_at
        LIMIT 1000;
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

export async function bulkSetSnapshotStatus(client: Client, ids: number[], status: string) {
	return await client.queryObject(
		`UPDATE snapshot_schedule SET status = $2 WHERE id = ANY($1)`,
		[ids, status],
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

export async function getAllVideosWithoutActiveSnapshotSchedule(client: Client) {
	const query: string = `
		SELECT s.aid
		FROM bilibili_metadata s
		LEFT JOIN snapshot_schedule ss ON s.aid = ss.aid AND (ss.status = 'pending' OR ss.status = 'processing')
		WHERE ss.aid IS NULL
	`;
	const res = await client.queryObject<{ aid: number }>(query, []);
	return res.rows.map((r) => Number(r.aid));
}
