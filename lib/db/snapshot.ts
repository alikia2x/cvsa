import { DAY, SECOND } from "$std/datetime/constants.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { VideoSnapshotType } from "lib/db/schema.d.ts";
import { parseTimestampFromPsql } from "lib/utils/formatTimestampToPostgre.ts";

export async function getVideosNearMilestone(client: Client) {
	const queryResult = await client.queryObject<VideoSnapshotType>(`
    	WITH filtered_snapshots AS (
			SELECT
				vs.*
			FROM
				video_snapshot vs
			WHERE
				(vs.views >= 90000 AND vs.views < 100000) OR
				(vs.views >= 900000 AND vs.views < 1000000)
		),
		ranked_snapshots AS (
			SELECT
				fs.*,
				ROW_NUMBER() OVER (PARTITION BY fs.aid ORDER BY fs.created_at DESC) as rn,
				MAX(fs.views) OVER (PARTITION BY fs.aid) as max_views_per_aid
			FROM
				filtered_snapshots fs
			INNER JOIN
				songs s ON fs.aid = s.aid
		)
		SELECT
			rs.id, rs.created_at, rs.views, rs.coins, rs.likes, rs.favorites, rs.shares, rs.danmakus, rs.aid, rs.replies
		FROM
			ranked_snapshots rs
		WHERE
			rs.rn = 1;
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
	const queryResult = await client.queryObject<{ count: number }>(
		`
		SELECT COUNT(*) AS count
		FROM video_snapshot
		WHERE aid = $1;
	`,
		[aid],
	);
	return queryResult.rows[0].count;
}

export async function getShortTermEtaPrediction(client: Client, aid: number) {
	const queryResult = await client.queryObject<{ eta: number }>(
		`
		WITH old_snapshot AS (
			SELECT created_at, views 
			FROM video_snapshot 
			WHERE aid = $1 AND
			NOW() - created_at > '20 min'
			ORDER BY created_at DESC 
			LIMIT 1
		),
		new_snapshot AS (
			SELECT created_at, views
			FROM video_snapshot
			WHERE aid = $1
			ORDER BY created_at DESC
			LIMIT 1
		)
		SELECT
			CASE 
				WHEN n.views > 100000 
				THEN 
					(1000000 - n.views) -- Views remaining
					/ 
					(
						(n.views - o.views)  -- Views delta
						/
						(EXTRACT(EPOCH FROM (n.created_at - o.created_at)) + 0.001) -- Time delta in seconds
					    + 0.001
					) -- Increment per second
				ELSE 
					(100000 - n.views) -- Views remaining
			        /
					(
					    (n.views - o.views) -- Views delta
			            /
					    (EXTRACT(EPOCH FROM (n.created_at - o.created_at)) + 0.001) -- Time delta in seconds
					    + 0.001
			        ) -- Increment per second
			END AS eta
		FROM old_snapshot o, new_snapshot n;
		`,
		[aid],
	);
	if (queryResult.rows.length === 0) {
		return null;
	}
	return queryResult.rows[0].eta;
}

export async function getIntervalFromLastSnapshotToNow(client: Client, aid: number) {
	const queryResult = await client.queryObject<{ interval: number }>(
		`
		SELECT EXTRACT(EPOCH FROM (NOW() - created_at)) AS interval
		FROM video_snapshot
		WHERE aid = $1
		ORDER BY created_at DESC
		LIMIT 1;
		`,
		[aid],
	);
	if (queryResult.rows.length === 0) {
		return null;
	}
	return queryResult.rows[0].interval;
}

export async function songEligibleForMilestoneSnapshot(client: Client, aid: number) {
	const count = await getSongSnapshotCount(client, aid);
	if (count < 2) {
		return true;
	}
	const queryResult = await client.queryObject<
		{ views1: number; created_at1: string; views2: number; created_at2: string }
	>(
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
