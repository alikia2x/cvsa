import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { aidExistsInSongs, getNotCollectedSongs } from "lib/db/songs.ts";
import logger from "lib/log/logger.ts";

export async function collectSongs(client: Client) {
	const aids = await getNotCollectedSongs(client);
	for (const aid of aids) {
		const exists = await aidExistsInSongs(client, aid);
		if (exists) continue;
		await insertIntoSongs(client, aid);
		logger.log(`Video ${aid} was added into the songs table.`, "mq", "fn:collectSongs");
	}
}

export async function insertIntoSongs(client: Client, aid: number) {
	await client.queryObject(
		`
			INSERT INTO songs (aid, bvid, published_at, duration)
			VALUES (
				$1,
				(SELECT bvid FROM all_data WHERE aid = $1),
				(SELECT published_at FROM all_data WHERE aid = $1),
				(SELECT duration FROM all_data WHERE aid = $1)
			)
			ON CONFLICT DO NOTHING
		`,
		[aid],
	);
}
