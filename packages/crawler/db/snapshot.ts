import { SnapshotNumber } from "mq/task/getVideoStats";
import type { Psql } from "@core/db/psql.d";
import { db, LatestVideoSnapshotType, videoSnapshot, VideoSnapshotType } from "@core/drizzle";
import { PartialBy } from "@core/lib";
import { sql } from "drizzle-orm";

export async function insertVideoSnapshot(data: PartialBy<VideoSnapshotType, "id">) {
	await db.insert(videoSnapshot).values(data);
}

export async function getVideosNearMilestone() {
	const results = await db.execute<LatestVideoSnapshotType>(sql`
		SELECT ls.*
		FROM latest_video_snapshot ls
			RIGHT JOIN songs ON songs.aid = ls.aid
		WHERE
			(views >= 60000 AND views < 100000) OR
			(views >= 900000 AND views < 1000000) OR
			views > 1000000
		UNION
		SELECT ls.*
		FROM latest_video_snapshot ls
		WHERE
			(views >= 90000 AND views < 100000) OR
			(views >= 900000 AND views < 1000000) OR
			(views >= CEIL(views::float/1000000::float)*1000000-100000 AND views < CEIL(views::float/1000000::float)*1000000)
		UNION
		SELECT ls.*
		FROM latest_video_snapshot ls
		JOIN eta ON eta.aid = ls.aid
		WHERE eta.eta < 2300
	`);

	return results.map((row) => {
		return {
			...row,
			aid: Number(row.aid)
		};
	});
}

export async function getLatestVideoSnapshot(
	sql: Psql,
	aid: number
): Promise<null | SnapshotNumber> {
	const queryResult = await sql<LatestVideoSnapshotType[]>`
	    SELECT *
	    FROM latest_video_snapshot
	    WHERE aid = ${aid}
	`;
	if (queryResult.length === 0) {
		return null;
	}
	return queryResult.map((row) => {
		return {
			...row,
			aid: Number(row.aid),
			time: new Date(row.time).getTime()
		};
	})[0];
}
