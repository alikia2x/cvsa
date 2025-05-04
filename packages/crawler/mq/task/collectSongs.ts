import { sql } from "@core/db/dbNew";
import { aidExistsInSongs, getNotCollectedSongs } from "db/songs.ts";
import logger from "@core/log/logger.ts";
import { scheduleSnapshot } from "db/snapshotSchedule.ts";
import { MINUTE } from "@core/const/time.ts";
import type { Psql } from "global.d.ts";

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
		INSERT INTO songs (aid, published_at, duration)
		VALUES (
			$1,
			(SELECT published_at FROM bilibili_metadata WHERE aid = ${aid}),
			(SELECT duration FROM bilibili_metadata WHERE aid = ${aid})
		)
		ON CONFLICT DO NOTHING
	`
}
