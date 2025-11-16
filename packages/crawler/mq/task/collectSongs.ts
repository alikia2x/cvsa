import { sql } from "@core/db/dbNew";
import { aidExistsInSongs, getNotCollectedSongs } from "db/songs";
import logger from "@core/log";
import { scheduleSnapshot } from "db/snapshotSchedule";
import { MINUTE } from "@core/lib";
import type { Psql } from "@core/db/psql.d";
import { db, songs } from "@core/drizzle";
import { and, eq, sql as drizzleSQL } from "drizzle-orm";

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
	const data = await db
		.insert(songs)
		.values({
			aid,
			publishedAt: drizzleSQL`SELECT published_at FROM bilibili_metadata WHERE aid = ${aid}`,
			duration: drizzleSQL`SELECT duration FROM bilibili_metadata WHERE aid = ${aid}`,
			image: drizzleSQL`SELECT cover_url FROM bilibili_metadata WHERE aid = ${aid}`,
			producer: drizzleSQL`
			SELECT username
			FROM bilibili_user bu
			JOIN bilibili_metadata bm
			ON bm.uid = bu.uid
			WHERE bm.aid = ${aid}
		`
		})
		.onConflictDoNothing()
		.returning();

	return data;
}
