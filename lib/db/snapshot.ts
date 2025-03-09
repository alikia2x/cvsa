import { DAY, SECOND } from "$std/datetime/constants.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { VideoSnapshotType } from "lib/db/schema.d.ts";
import { parseTimestampFromPsql } from "lib/utils/formatTimestampToPostgre.ts";

export async function getSongsNearMilestone(client: Client) {
	const queryResult = await client.queryObject<VideoSnapshotType>(`
    	WITH max_views_per_aid AS (
			-- 找出每个 aid 的最大 views 值，并确保 aid 存在于 songs 表中
			SELECT 
				vs.aid, 
				MAX(vs.views) AS max_views
			FROM 
				video_snapshot vs
			INNER JOIN 
				songs s
			ON 
				vs.aid = s.aid
			GROUP BY 
				vs.aid
		),
		filtered_max_views AS (
			-- 筛选出满足条件的最大 views
			SELECT 
				aid, 
				max_views
			FROM 
				max_views_per_aid
			WHERE 
				(max_views >= 90000 AND max_views < 100000) OR
				(max_views >= 900000 AND max_views < 1000000)
		)
		-- 获取符合条件的完整行数据
		SELECT 
			vs.*
		FROM 
			video_snapshot vs
		INNER JOIN 
			filtered_max_views fmv
		ON 
			vs.aid = fmv.aid AND vs.views = fmv.max_views
    `);
	return queryResult.rows.map((row) => {
		return {
			...row,
			aid: Number(row.aid),
		};
	});
}

export async function getUnsnapshotedSongs(client: Client) {
	const queryResult = await client.queryObject<{ aid: bigint }>(`
		SELECT DISTINCT s.aid
		FROM songs s
		LEFT JOIN video_snapshot v ON s.aid = v.aid
		WHERE v.aid IS NULL;
	`);
	return queryResult.rows.map((row) => Number(row.aid));
}

export async function getSongSnapshotCount(client: Client, aid: number) {
	const queryResult = await client.queryObject<{ count: number }>(`
		SELECT COUNT(*) AS count
		FROM video_snapshot
		WHERE aid = $1;
	`, [aid]);
	return queryResult.rows[0].count;
}

export async function songEligibleForMilestoneSnapshot(client: Client, aid: number) {
	const count = await getSongSnapshotCount(client, aid);
	if (count < 2) {
		return true;
	}
	const queryResult = await client.queryObject<{ views1: number, created_at1: string, views2: number, created_at2: string }>(
		`
			WITH latest_snapshot AS (
				SELECT 
					aid,
					views,
					created_at
				FROM video_snapshot
				WHERE aid = $1
				ORDER BY created_at DESC
				LIMIT 1
			),
			pairs AS (
				SELECT 
					a.views AS views1,
					a.created_at AS created_at1,
					b.views AS views2,
					b.created_at AS created_at2,
					(b.created_at - a.created_at) AS interval
				FROM video_snapshot a
				JOIN latest_snapshot b 
					ON a.aid = b.aid 
					AND a.created_at < b.created_at
			)
			SELECT 
				views1,
				created_at1,
				views2,
				created_at2
			FROM (
				SELECT 
					*,
					ROW_NUMBER() OVER (
						ORDER BY 
							CASE WHEN interval <= INTERVAL '3 days' THEN 0 ELSE 1 END,
							CASE WHEN interval <= INTERVAL '3 days' THEN -interval ELSE interval END
					) AS rn
				FROM pairs
			) ranked
			WHERE rn = 1;
		`,
		[aid],
	);
	if (queryResult.rows.length === 0) {
		return true;
	}
	const recentViewsData = queryResult.rows[0];
	const time1 = parseTimestampFromPsql(recentViewsData.created_at1);
	const time2 = parseTimestampFromPsql(recentViewsData.created_at2);
	const intervalSec = (time2 - time1) / SECOND;
	const views1 = recentViewsData.views1;
	const views2 = recentViewsData.views2;
	const viewsDiff = views2 - views1;
	if (viewsDiff == 0) {
		return false;
	}
	const nextMilestone = views2 >= 100000 ? 1000000 : 100000;
	const expectedViewsDiff = nextMilestone - views2;
	const expectedIntervalSec = expectedViewsDiff / viewsDiff * intervalSec;
	return expectedIntervalSec <= 3 * DAY;
}
