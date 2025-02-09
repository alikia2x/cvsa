import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { AllDataType } from "lib/db/schema.d.ts";
import logger from "lib/log/logger.ts";

export async function videoExistsInAllData(client: Client, aid: number) {
	return await client.queryObject<{ exists: boolean }>("SELECT EXISTS(SELECT 1 FROM all_data WHERE aid = $1)", [aid])
		.then((result) => result.rows[0].exists);
}

export async function insertIntoAllData(client: Client, data: AllDataType) {
    logger.log(`inserted ${data.aid}`, "db-all_data")
	return await client.queryObject(
		"INSERT INTO all_data (aid, bvid, description, uid, tags, title, published_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
		[data.aid, data.bvid, data.description, data.uid, data.tags, data.title, data.published_at],
	);
}

export async function getLatestVideoTimestampFromAllData(client: Client) {
	return await client.queryObject<{ published_at: string }>("SELECT published_at FROM all_data ORDER BY published_at DESC LIMIT 1")
		.then((result) => {
			const date = new Date(result.rows[0].published_at);
			if (isNaN(date.getTime())) {
				return null;
			}
			return date.getTime();
		});
}
