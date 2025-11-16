import { LatestSnapshotType } from "@core/db/schema";
import { SnapshotNumber } from "mq/task/getVideoStats";
import type { Psql } from "@core/db/psql.d";
import {
	db,
	eta,
	latestVideoSnapshot as lv,
	songs,
	videoSnapshot,
	VideoSnapshotType
} from "@core/drizzle";
import { PartialBy } from "@core/lib";
import { and, eq, getTableColumns, gte, lte, lt, or } from "drizzle-orm";
import { union } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export async function insertVideoSnapshot(data: PartialBy<VideoSnapshotType, "id">) {
	await db.insert(videoSnapshot).values(data);
}

export async function getVideosNearMilestone() {
	const results = await union(
		db
			.select({ ...getTableColumns(lv) })
			.from(lv)
			.rightJoin(songs, eq(lv.aid, songs.aid))
			.where(
				or(
					and(gte(lv.views, 60000), lt(lv.views, 100000)),
					and(gte(lv.views, 900000), lt(lv.views, 1000000)),
					gte(lv.views, 1000000)
				)
			),
		db
			.select({ ...getTableColumns(lv) })
			.from(lv)
			.where(
				or(
					and(gte(lv.views, 60000), lt(lv.views, 100000)),
					and(gte(lv.views, 900000), lt(lv.views, 1000000)),
					and(
						sql`views >= CEIL(views::float/1000000::float)*1000000-100000`,
						sql`views < CEIL(views::float/1000000::float)*1000000)`
					)
				)
			),
		db
			.select({ ...getTableColumns(lv) })
			.from(lv)
			.innerJoin(eta, eq(lv.aid, eta.aid))
			.where(lte(eta.eta, 2300))
	);
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
	const queryResult = await sql<LatestSnapshotType[]>`
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
