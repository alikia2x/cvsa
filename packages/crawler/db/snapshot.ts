import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { LatestSnapshotType } from "@core/db/schema";
import { SnapshotNumber } from "mq/task/getVideoStats.ts";

export async function getVideosNearMilestone(client: Client) {
	const queryResult = await client.queryObject<LatestSnapshotType>(`
        SELECT ls.*
        FROM latest_video_snapshot ls
        WHERE
            (views >= 5000 AND views < 100000) OR
            (views >= 900000 AND views < 1000000) OR
            (views >= 9900000 AND views < 10000000)
    `);
	return queryResult.rows.map((row) => {
		return {
			...row,
			aid: Number(row.aid),
		};
	});
}

export async function getLatestVideoSnapshot(client: Client, aid: number): Promise<null | SnapshotNumber> {
	const queryResult = await client.queryObject<LatestSnapshotType>(
		`
	    SELECT *
	    FROM latest_video_snapshot
	    WHERE aid = $1
	`,
		[aid],
	);
	if (queryResult.rows.length === 0) {
		return null;
	}
	return queryResult.rows.map((row) => {
		return {
			...row,
			aid: Number(row.aid),
			time: new Date(row.time).getTime(),
		};
	})[0];
}
