import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { VideoSnapshotType } from "lib/db/schema.d.ts";

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
		}
	});
}

export async function getUnsnapshotedSongs(client: Client) {
	const queryResult = await client.queryObject<{aid: bigint}>(`
		SELECT DISTINCT s.aid
		FROM songs s
		LEFT JOIN video_snapshot v ON s.aid = v.aid
		WHERE v.aid IS NULL;
	`);
	return queryResult.rows.map(row => Number(row.aid));
}
