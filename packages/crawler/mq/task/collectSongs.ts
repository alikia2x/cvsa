import { sql } from "@core/db/dbNew";
import { aidExistsInSongs, getNotCollectedSongs } from "db/songs";
import logger from "@core/log";
import { scheduleSnapshot } from "db/snapshotSchedule";
import { MINUTE } from "@core/lib";
import type { Psql } from "@core/db/psql.d";

export async function collectSongs() {
	const aids = await getNotCollectedSongs(sql);
	for (const aid of aids) {
		const exists = await aidExistsInSongs(sql, aid);
		if (exists) continue;
		await insertIntoSongs(sql, aid);
		await scheduleSnapshot(sql, aid, "new", Date.now() + 10 * MINUTE, true);
		logger.log(`Video ${aid} was added into the songs table.`, "mq", "fn:collectSongs");
	}
}

export async function insertIntoSongs(sql: Psql, aid: number) {
	await sql`
		INSERT INTO songs (aid, published_at, duration, image, producer)
		VALUES (
			$1,
			(SELECT published_at FROM bilibili_metadata WHERE aid = ${aid}),
			(SELECT duration FROM bilibili_metadata WHERE aid = ${aid}),
			(SELECT cover_url FROM bilibili_metadata WHERE aid = ${aid}),
			(
				SELECT username
				FROM bilibili_user bu
				JOIN bilibili_metadata bm
				ON bm.uid = bu.uid
				WHERE bm.aid = ${aid}
			)
		)
		ON CONFLICT DO NOTHING
	`;
}
