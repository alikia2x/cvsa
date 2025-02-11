import { Client, Transaction } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { AllDataType } from "lib/db/schema.d.ts";
import logger from "lib/log/logger.ts";
import { parseTimestampFromPsql } from "lib/utils/formatTimestampToPostgre.ts";

export async function videoExistsInAllData(client: Client, aid: number) {
	return await client.queryObject<{ exists: boolean }>(`SELECT EXISTS(SELECT 1 FROM all_data WHERE aid = $1)`, [aid])
		.then((result) => result.rows[0].exists);
}

export async function insertIntoAllData(client: Client, data: AllDataType) {
	logger.log(`inserted ${data.aid}`, "db-all_data");
	return await client.queryObject(
		`INSERT INTO all_data (aid, bvid, description, uid, tags, title, published_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (aid) DO NOTHING`,
		[data.aid, data.bvid, data.description, data.uid, data.tags, data.title, data.published_at],
	);
}

export async function getLatestVideoTimestampFromAllData(client: Client) {
	return await client.queryObject<{ published_at: string }>(
		`SELECT published_at FROM all_data ORDER BY published_at DESC LIMIT 1`,
	)
		.then((result) => {
			const date = new Date(result.rows[0].published_at);
			if (isNaN(date.getTime())) {
				return null;
			}
			return date.getTime();
		});
}

export async function videoTagsIsNull(client: Client | Transaction, aid: number) {
	return await client.queryObject<{ exists: boolean }>(
		`SELECT EXISTS(SELECT 1 FROM all_data WHERE aid = $1 AND tags IS NULL)`,
		[aid],
	).then((result) => result.rows[0].exists);
}

export async function updateVideoTags(client: Client | Transaction, aid: number, tags: string[]) {
	return await client.queryObject(
		`UPDATE all_data SET tags = $1 WHERE aid = $2`,
		[tags.join(","), aid],
	);
}

export async function getNullVideoTagsList(client: Client) {
	const queryResult = await client.queryObject<{ aid: number; published_at: string }>(
		`SELECT aid, published_at FROM all_data WHERE tags IS NULL`,
	);
	const rows = queryResult.rows;
	return rows.map(
		(row) => {
			return {
				aid: Number(row.aid),
				published_at: parseTimestampFromPsql(row.published_at),
			};
		},
	);
}
