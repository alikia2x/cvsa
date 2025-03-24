import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { aidExistsInSongs, getNotCollectedSongs } from "lib/db/songs.ts";
import logger from "lib/log/logger.ts";
import { scheduleSnapshot } from "lib/db/snapshotSchedule.ts";
import { MINUTE } from "$std/datetime/constants.ts";

export async function collectSongs(client: Client) {
	const aids = await getNotCollectedSongs(client);
	for (const aid of aids) {
		const exists = await aidExistsInSongs(client, aid);
		if (exists) continue;
		await insertIntoSongs(client, aid);
		await scheduleSnapshot(client, aid, "new", Date.now() + 10 * MINUTE, true);
		logger.log(`Video ${aid} was added into the songs table.`, "mq", "fn:collectSongs");
	}
}

export async function insertIntoSongs(client: Client, aid: number) {
	await client.queryObject(
		`
			INSERT INTO songs (aid, published_at, duration)
			VALUES (
				$1,
				(SELECT published_at FROM bilibili_metadata WHERE aid = $1),
				(SELECT duration FROM bilibili_metadata WHERE aid = $1)
			)
			ON CONFLICT DO NOTHING
		`,
		[aid],
	);
}
