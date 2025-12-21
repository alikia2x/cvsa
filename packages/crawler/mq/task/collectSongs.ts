import { sql } from "@core/db/dbNew";
import type { Psql } from "@core/db/psql.d";
import { db, songs } from "@core/drizzle";
import { MINUTE } from "@core/lib";
import logger from "@core/log";
import { scheduleSnapshot } from "db/snapshotSchedule";
import { aidExistsInSongs, getNotCollectedSongs } from "db/songs";
import { and, sql as drizzleSQL, eq } from "drizzle-orm";

export async function collectSongs() {
	const aids = await getNotCollectedSongs(sql);
	for (const aid of aids) {
		const exists = await aidExistsInSongs(sql, aid);
		if (exists) continue;
		await insertIntoSongs(aid);
		await scheduleSnapshot(sql, aid, "new", Date.now() + 10 * MINUTE, true);
		logger.log(`Video ${aid} was added into the songs table.`, "mq", "fn:collectSongs");
	}
}

export async function insertIntoSongs(aid: number) {
	const song = await db
		.select({ id: songs.id })
		.from(songs)
		.where(and(eq(songs.aid, aid), eq(songs.deleted, true)))
		.limit(1);
	const songExistsAndDeleted = song.length > 0;

	if (songExistsAndDeleted) {
		const data = await db
			.update(songs)
			.set({ deleted: false })
			.where(eq(songs.id, song[0].id))
			.returning();
		return data;
	}
	const data = await sql`
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
		RETURNING *
	`;

	return data;
}
