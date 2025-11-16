import type { SnapshotScheduleType } from "@core/db/schema.d";
import logger from "@core/log";
import { MINUTE } from "@core/lib";
import { redis } from "@core/db/redis";
import { Redis } from "ioredis";
import { parseTimestampFromPsql } from "../utils/formatTimestampToPostgre";
import type { Psql } from "@core/db/psql.d";

const REDIS_KEY = "cvsa:snapshot_window_counts";

function getCurrentWindowIndex(): number {
	const now = new Date();
	const minutesSinceMidnight = now.getHours() * 60 + now.getMinutes();
	return Math.floor(minutesSinceMidnight / 5);
}

export async function refreshSnapshotWindowCounts(sql: Psql, redisClient: Redis) {
	const now = new Date();
	const startTime = now.getTime();

	const result = await sql<{ window_start: Date; count: number }[]>`
		SELECT
		date_trunc('hour', started_at) +
		(EXTRACT(minute FROM started_at)::int / 5 * INTERVAL '5 minutes') AS window_start,
		COUNT(*) AS count
		FROM snapshot_schedule
		WHERE started_at >= NOW() AND status = 'pending' AND started_at <= NOW() + INTERVAL '14 days'
		GROUP BY 1
		ORDER BY window_start
  	`;

	await redisClient.del(REDIS_KEY);

	const currentWindow = getCurrentWindowIndex();

	for (const row of result) {
		const targetOffset = Math.floor((row.window_start.getTime() - startTime) / (5 * MINUTE));
		const offset = currentWindow + targetOffset;
		if (offset >= 0) {
			await redisClient.hset(REDIS_KEY, offset.toString(), Number(row.count));
		}
	}
}

export async function initSnapshotWindowCounts(sql: Psql, redisClient: Redis) {
	await refreshSnapshotWindowCounts(sql, redisClient);
	setInterval(async () => {
		await refreshSnapshotWindowCounts(sql, redisClient);
	}, 5 * MINUTE);
}

async function getWindowCount(redisClient: Redis, offset: number): Promise<number> {
	const count = await redisClient.hget(REDIS_KEY, offset.toString());
	return count ? parseInt(count, 10) : 0;
}

export async function snapshotScheduleExists(sql: Psql, id: number) {
	const rows = await sql<{ id: number }[]>`
		SELECT id
		FROM snapshot_schedule
		WHERE id = ${id}
	`;
	return rows.length > 0;
}

export async function videoHasActiveScheduleWithType(sql: Psql, aid: number, type: string) {
	const rows = await sql<{ status: string }[]>`
		SELECT status FROM snapshot_schedule
		WHERE aid = ${aid}
			AND (status = 'pending' OR status = 'processing')
			AND type = ${type}
	`;
	return rows.length > 0;
}

export async function videoHasProcessingSchedule(sql: Psql, aid: number) {
	const rows = await sql<{ status: string }[]>`
		SELECT status
		FROM snapshot_schedule
		WHERE aid = ${aid}
			AND status = 'processing'
	`;
	return rows.length > 0;
}

export async function bulkGetVideosWithoutProcessingSchedules(sql: Psql, aids: number[]) {
	const rows = await sql<{ aid: string }[]>`
		SELECT aid
		FROM snapshot_schedule
		WHERE aid = ANY(${aids})
			AND status != 'processing'
		GROUP BY aid
	`;
	return rows.map((row) => Number(row.aid));
}

interface Snapshot {
	created_at: number;
	views: number;
}

export async function findClosestSnapshot(
	sql: Psql,
	aid: number,
	targetTime: Date
): Promise<Snapshot | null> {
	const result = await sql<{ created_at: string; views: number }[]>`
		WITH target_time AS NOT MATERIALIZED (
			SELECT ${targetTime.toISOString()}::timestamptz AS t
		),
		before AS NOT MATERIALIZED (
			SELECT 
				created_at, 
				views,
				EXTRACT(EPOCH FROM (t - created_at)) AS distance
			FROM video_snapshot, target_time
			WHERE aid = ${aid} 
			AND created_at <= t
			ORDER BY created_at DESC
			LIMIT 1
		),
		after AS NOT MATERIALIZED (
			SELECT 
				created_at, 
				views,
				EXTRACT(EPOCH FROM (created_at - t)) AS distance
			FROM video_snapshot, target_time
			WHERE aid = ${aid} 
			AND created_at >= t
			ORDER BY created_at ASC
			LIMIT 1
		)
		SELECT created_at, views
		FROM (
			SELECT *, ROW_NUMBER() OVER (ORDER BY distance) AS rn
			FROM (SELECT * FROM before UNION ALL SELECT * FROM after) AS combined
		) AS ranked
		WHERE rn = 1;
	`;
	if (result.length === 0) return null;
	const row = result[0];
	return {
		created_at: new Date(row.created_at).getTime(),
		views: row.views
	};
}

export async function findSnapshotBefore(
	sql: Psql,
	aid: number,
	targetTime: Date
): Promise<Snapshot | null> {
	const result = await sql<{ created_at: string; views: number }[]>`
        SELECT created_at, views
        FROM video_snapshot
        WHERE aid = ${aid}
		AND created_at <= ${targetTime.toISOString()}::timestamptz
        ORDER BY created_at DESC
        LIMIT 1
	`;
	if (result.length === 0) return null;
	const row = result[0];
	return {
		created_at: new Date(row.created_at).getTime(),
		views: row.views
	};
}

export async function hasAtLeast2Snapshots(sql: Psql, aid: number) {
	const res = await sql<{ exists: boolean }[]>`
	  SELECT EXISTS (
		SELECT 1 
		FROM video_snapshot 
		WHERE aid = ${aid}
		LIMIT 2
	  ) AS exists
	`;
	return res[0].exists;
}

export async function getLatestSnapshot(sql: Psql, aid: number): Promise<Snapshot | null> {
	const res = await sql<{ created_at: string; views: number }[]>`
		SELECT created_at, views
		FROM video_snapshot
		WHERE aid = ${aid}
		ORDER BY created_at DESC
		LIMIT 1
	`;
	if (res.length === 0) return null;
	const row = res[0];
	return {
		created_at: new Date(row.created_at).getTime(),
		views: row.views
	};
}

export async function getLatestActiveScheduleWithType(sql: Psql, aid: number, type: string) {
	const rows = await sql`
		SELECT *
		FROM snapshot_schedule
		WHERE aid = ${aid}
		  AND type = ${type}
		  AND (status = 'pending' OR status = 'processing')
		ORDER BY started_at DESC
		LIMIT 1
	`;
	return rows.length > 0 ? rows[0] : null;
}

/*
 * Creates a new snapshot schedule record.
 * @param aid The aid of the video.
 * @param type Type of the snapshot.
 * @param targetTime Scheduled time for snapshot. (Timestamp in milliseconds)
 * @param force Ignore all restrictions and force the creation of the schedule.
 */
export async function scheduleSnapshot(
	sql: Psql,
	aid: number,
	type: string,
	targetTime: number,
	force: boolean = false,
	adjustTime: boolean = true
) {
	let adjustedTime = new Date(targetTime);
	const hasActiveSchedule = await videoHasActiveScheduleWithType(sql, aid, type);
	if (type == "milestone" && hasActiveSchedule) {
		const latestActiveSchedule = await getLatestActiveScheduleWithType(sql, aid, type);
		if (!latestActiveSchedule) {
			return;
		}
		const latestScheduleStartedAt = new Date(
			parseTimestampFromPsql(latestActiveSchedule.started_at)
		);
		if (latestScheduleStartedAt > adjustedTime) {
			await sql`
                UPDATE snapshot_schedule
                SET started_at = ${adjustedTime}
                WHERE id = ${latestActiveSchedule.id}
			`;
			logger.log(
				`Updated snapshot schedule for ${aid} at ${adjustedTime.toISOString()}`,
				"mq",
				"fn:scheduleSnapshot"
			);
			return;
		}
	}
	if (hasActiveSchedule && !force) return;
	if (type !== "milestone" && type !== "new" && adjustTime) {
		adjustedTime = await adjustSnapshotTime(new Date(targetTime), 3000, redis);
	}
	logger.log(
		`Scheduled snapshot for ${aid} at ${adjustedTime.toISOString()}`,
		"mq",
		"fn:scheduleSnapshot"
	);
	return sql`
		INSERT INTO snapshot_schedule
			(aid, type, started_at)
			VALUES (
				${aid},
				${type},
				${adjustedTime.toISOString()}
			)
	`;
}

export async function bulkScheduleSnapshot(
	sql: Psql,
	aids: number[],
	type: string,
	targetTime: number,
	force: boolean = false,
	adjustTime: boolean = true
) {
	for (const aid of aids) {
		await scheduleSnapshot(sql, aid, type, targetTime, force, adjustTime);
	}
}

export async function adjustSnapshotTime(
	expectedStartTime: Date,
	allowedCounts: number = 1000,
	redisClient: Redis
): Promise<Date> {
	const currentWindow = getCurrentWindowIndex();
	const targetOffset = Math.floor((expectedStartTime.getTime() - Date.now()) / (5 * MINUTE)) - 6;

	const initialOffset = currentWindow + Math.max(targetOffset, 0);

	const MAX_ITERATIONS = 4032; // 10 days
	for (let i = initialOffset; i < MAX_ITERATIONS; i++) {
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
				return now;
			}
			return delayedDate;
		}
	}
	return expectedStartTime;
}

export async function getSnapshotsInNextSecond(sql: Psql) {
	return sql<SnapshotScheduleType[]>`
        SELECT *
        FROM snapshot_schedule
        WHERE started_at <= NOW() + INTERVAL '1 seconds'
		  AND started_at >= NOW() - INTERVAL '1 minutes'
          AND status = 'pending'
          AND type != 'normal'
        LIMIT 10;
	`;
}

export async function getBulkSnapshotsInNextSecond(sql: Psql) {
	return sql<SnapshotScheduleType[]>`
        SELECT *
        FROM snapshot_schedule
        WHERE (started_at <= NOW() + INTERVAL '15 seconds')
		  AND started_at >= NOW() - INTERVAL '2 minutes'
          AND status = 'pending'
          AND (type = 'normal' OR type = 'archive')
        ORDER BY CASE
                     WHEN type = 'normal' THEN 1
                     WHEN type = 'archive' THEN 2
                     END,
                 started_at
        LIMIT 1000;
	`;
}

export async function setSnapshotStatus(sql: Psql, id: number, status: string) {
	return sql`
        UPDATE snapshot_schedule
        SET status = ${status}
        WHERE id = ${id}
	`;
}

export async function bulkSetSnapshotStatus(sql: Psql, ids: number[], status: string) {
	return sql`
        UPDATE snapshot_schedule
        SET status = ${status}
        WHERE id = ANY (${ids})
	`;
}

export async function getVideosWithoutActiveSnapshotScheduleByType(sql: Psql, type: string) {
	const rows = await sql<{ aid: string }[]>`
		SELECT s.aid
		FROM songs s
		LEFT JOIN snapshot_schedule ss ON
		    s.aid = ss.aid AND
		    (ss.status = 'pending' OR ss.status = 'processing') AND
		    ss.type = ${type}
		WHERE ss.aid IS NULL
	`;
	return rows.map((r) => Number(r.aid));
}

export async function getCommonArchiveAids(sql: Psql) {
	const rows = await sql<{ aid: string }[]>`
		SELECT b.aid
		FROM bilibili_metadata b
		LEFT JOIN snapshot_schedule ss ON
			b.aid = ss.aid AND
			(ss.status = 'pending' OR ss.status = 'processing') AND
			ss.type = 'archive'
		WHERE ss.aid IS NULL
		AND NOT EXISTS (
			SELECT 1
			FROM songs s
			WHERE s.aid = b.aid
		)
	`;
	return rows.map((r) => Number(r.aid));
}
