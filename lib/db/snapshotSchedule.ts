import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { formatTimestampToPsql } from "lib/utils/formatTimestampToPostgre.ts";
import { SnapshotScheduleType } from "./schema.d.ts";
import logger from "lib/log/logger.ts";
import { DAY, MINUTE } from "$std/datetime/constants.ts";
import { redis } from "lib/db/redis.ts";
import { Redis } from "ioredis";

const WINDOW_SIZE = 2880; // 每天 2880 个 5 分钟窗口
const REDIS_KEY = "cvsa:snapshot_window_counts"; // Redis Key 名称

// 获取当前时间对应的窗口索引
function getCurrentWindowIndex(): number {
	const now = new Date();
	const minutesSinceMidnight = now.getHours() * 60 + now.getMinutes();
	const currentWindow = Math.floor(minutesSinceMidnight / 5);
	return currentWindow;
}

// 刷新内存数组
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
  	`

	await redisClient.del(REDIS_KEY);

	for (const row of result.rows) {
		const offset = Math.floor((row.window_start.getTime() - startTime) / (5 * MINUTE));
		if (offset >= 0 && offset < WINDOW_SIZE) {
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

async function updateWindowCount(redisClient: Redis, offset: number, increment: number): Promise<void> {
	await redisClient.hincrby(REDIS_KEY, offset.toString(), increment);
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
	let adjustedTime = new Date(targetTime);
	if (type !== "milestone") {
		adjustedTime = await adjustSnapshotTime(new Date(targetTime), 1000, redis);
	}
	logger.log(`Scheduled snapshot for ${aid} at ${adjustedTime.toISOString()}`, "mq", "fn:scheduleSnapshot");
	return client.queryObject(
		`INSERT INTO snapshot_schedule (aid, type, started_at) VALUES ($1, $2, $3)`,
		[aid, type, adjustedTime.toISOString()],
	);
}

export async function adjustSnapshotTime(
	expectedStartTime: Date,
	allowedCounts: number = 1000,
	redisClient: Redis,
): Promise<Date> {
	const currentWindow = getCurrentWindowIndex();

	// 计算目标窗口偏移量
	const targetOffset = Math.floor((expectedStartTime.getTime() - Date.now()) / (5 * 60 * 1000));

	// 在 Redis 中查找可用窗口
	for (let i = 0; i < WINDOW_SIZE; i++) {
		const offset = (currentWindow + targetOffset + i) % WINDOW_SIZE;
		const count = await getWindowCount(redisClient, offset);

		if (count < allowedCounts) {
			// 找到可用窗口，更新计数
			await updateWindowCount(redisClient, offset, 1);

			// 计算具体时间
			const windowStart = new Date(Date.now() + offset * 5 * 60 * 1000);
			const randomDelay = Math.floor(Math.random() * 5 * 60 * 1000);
			return new Date(windowStart.getTime() + randomDelay);
		}
	}

	// 如果没有找到可用窗口，返回原始时间
	return expectedStartTime;
}

export async function cleanupExpiredWindows(redisClient: Redis): Promise<void> {
	const now = new Date();
	const startTime = new Date(now.getTime() - 10 * DAY); // 保留最近 10 天的数据

	// 获取所有窗口索引
	const allOffsets = await redisClient.hkeys(REDIS_KEY);

	// 删除过期窗口
	for (const offsetStr of allOffsets) {
		const offset = parseInt(offsetStr, 10);
		const windowStart = new Date(startTime.getTime() + offset * 5 * MINUTE);

		if (windowStart < startTime) {
			await redisClient.hdel(REDIS_KEY, offsetStr);
		}
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
