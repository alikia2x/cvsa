import type { Psql } from "@core/db/psql.d";
import { redis } from "@core/db/redis";
import type { SnapshotScheduleType } from "@core/db/schema.d";
import { db, snapshotSchedule } from "@core/drizzle";
import { MINUTE } from "@core/lib";
import logger from "@core/log";
import dayjs from "dayjs";
import { eq, inArray } from "drizzle-orm";
import type { Redis } from "ioredis";
import { parseTimestampFromPsql } from "../utils/formatTimestampToPostgre";

const REDIS_KEY = "cvsa:snapshot_window_counts";

const WINDOW_SIZE = 5 * MINUTE;

function getWindowFromDate(date: Date) {
	const roundedMs = Math.floor(date.getTime() / WINDOW_SIZE) * WINDOW_SIZE;
	return new Date(roundedMs);
}

export async function refreshSnapshotWindowCounts(sql: Psql, redisClient: Redis) {
	const result = await sql<{ window_start: Date; count: number }[]>`
		SELECT
			started_at_5min_utc AT TIME ZONE 'UTC' AS window_start,
			count
		FROM (
			SELECT
				started_at_5min_utc,
				COUNT(*) AS count
			FROM snapshot_schedule
			WHERE
				status = 'pending'
			AND started_at_5min_utc >= date_trunc('hour', now() AT TIME ZONE 'UTC')
			AND started_at_5min_utc <  date_trunc('hour', now() AT TIME ZONE 'UTC')
				+ interval '14 days'
			GROUP BY started_at_5min_utc
		) t
		ORDER BY started_at_5min_utc
  	`;

	await redisClient.del(REDIS_KEY);

	for (const row of result) {
		await redisClient.hset(REDIS_KEY, row.window_start.toISOString(), Number(row.count));
	}
}

export async function initSnapshotWindowCounts(sql: Psql, redisClient: Redis) {
	await refreshSnapshotWindowCounts(sql, redisClient);
	setInterval(async () => {
		await refreshSnapshotWindowCounts(sql, redisClient);
	}, 5 * MINUTE);
}

async function getWindowCount(redisClient: Redis, window: Date): Promise<number> {
	const count = await redisClient.hget(REDIS_KEY, window.toISOString());
	return count ? parseInt(count, 10) : 0;
}

async function incrWindowCount(redisClient: Redis, window: Date) {
	return redisClient.hincrby(REDIS_KEY, window.toISOString(), 1);
}

async function decrWindowCount(redisClient: Redis, window: Date) {
	return redisClient.hincrby(REDIS_KEY, window.toISOString(), -1);
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
		views: row.views,
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
		views: row.views,
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
		views: row.views,
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
	if (type === "milestone" && hasActiveSchedule) {
		const latestActiveSchedule = await getLatestActiveScheduleWithType(sql, aid, type);
		if (!latestActiveSchedule) {
			return;
		}
		const latestScheduleStartedAt = new Date(
			parseTimestampFromPsql(latestActiveSchedule.started_at)
		);
		if (latestScheduleStartedAt > adjustedTime) {
			await db.transaction(async (tx) => {
				const old = await tx.select().from(snapshotSchedule).where(
					eq(snapshotSchedule.id, latestActiveSchedule.id)
				).for("update");
				if (old.length === 0) return;
				const oldSchedule = old[0];
				await tx.update(snapshotSchedule).set({ startedAt: adjustedTime.toISOString() }).where(
					eq(snapshotSchedule.id, latestActiveSchedule.id)
				);
				if (oldSchedule.status !== "pending") return;
				const oldWindow = getWindowFromDate(new Date(oldSchedule.startedAt));
				await decrWindowCount(redis, oldWindow);
				const window = getWindowFromDate(adjustedTime);
				await incrWindowCount(redis, window);
			});
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
	else {
		const window = getWindowFromDate(adjustedTime);
    	await incrWindowCount(redis, window);
	}
	logger.log(
		`Scheduled ${type} snapshot for ${aid} at ${adjustedTime.toISOString()}`,
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
	const initialWindow = dayjs(getWindowFromDate(expectedStartTime));

	const MAX_ITERATIONS = 4032; // 10 days
	for (let i = 0; i < MAX_ITERATIONS; i++) {
		const window = initialWindow.add(i * WINDOW_SIZE, "milliseconds");
		const newCount = await incrWindowCount(redisClient, window.toDate());

		if (newCount > allowedCounts) {
			await decrWindowCount(redisClient, window.toDate());
			continue;
		}

		const randomDelay = Math.random() * WINDOW_SIZE;

		const randomizedExecutionTime = window.add(randomDelay, "milliseconds");
		const delayedDate = randomizedExecutionTime.toDate();
		const now = new Date();

		if (delayedDate.getTime() < now.getTime()) {
			return now;
		}
		return delayedDate;
	}

	for (let i = 0; i < 6; i++) {
		const window = initialWindow.subtract(i * WINDOW_SIZE, "milliseconds");

		const newCount = await incrWindowCount(redisClient, window.toDate());
		if (newCount <= allowedCounts) {
			return window.toDate();
		}
		await decrWindowCount(redisClient, window.toDate());
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

export async function setSnapshotStatus(_sql: Psql, id: number, status: string) {
	return await db.transaction(async (tx) => {
		const snapshots = await tx
			.select()
			.from(snapshotSchedule)
			.where(eq(snapshotSchedule.id, id))
			.for("update");

		if (snapshots.length === 0) return;

		const snapshot = snapshots[0];

		const removeFromPending = snapshot.status === "pending" && status !== "pending";

		if (removeFromPending) {
			const window = getWindowFromDate(new Date(snapshot.startedAt));
			await decrWindowCount(redis, window);
		}

		return await tx.update(snapshotSchedule).set({ status }).where(eq(snapshotSchedule.id, id));
	});
}

export async function bulkSetSnapshotStatus(_sql: Psql, ids: number[], status: string) {
	return await db.transaction(async (tx) => {
		const snapshots = await tx
			.select()
			.from(snapshotSchedule)
			.where(inArray(snapshotSchedule.id, ids))
			.for("update");

		if (snapshots.length === 0) return;

		for (const snapshot of snapshots) {
			const removeFromPending = snapshot.status === "pending" && status !== "pending";
			if (removeFromPending) {
				const window = getWindowFromDate(new Date(snapshot.startedAt));
				await decrWindowCount(redis, window);
			}
		}

		return await tx.update(snapshotSchedule).set({ status }).where(inArray(snapshotSchedule.id, ids));
	});
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
