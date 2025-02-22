import { Client, Transaction } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { AllDataType } from "lib/db/schema.d.ts";
import logger from "lib/log/logger.ts";
import { formatTimestampToPsql, parseTimestampFromPsql } from "lib/utils/formatTimestampToPostgre.ts";
import { VideoListVideo } from "lib/net/bilibili.d.ts";
import { HOUR, SECOND } from "$std/datetime/constants.ts";
import { modelVersion } from "lib/ml/filter_inference.ts";

export async function videoExistsInAllData(client: Client, aid: number) {
	return await client.queryObject<{ exists: boolean }>(`SELECT EXISTS(SELECT 1 FROM all_data WHERE aid = $1)`, [aid])
		.then((result) => result.rows[0].exists);
}

export async function biliUserExists(client: Client, uid: number) {
	return await client.queryObject<{ exists: boolean }>(`SELECT EXISTS(SELECT 1 FROM bili_user WHERE uid = $1)`, [uid])
		.then((result) => result.rows[0].exists);
}

export async function insertIntoAllData(client: Client, data: VideoListVideo) {
	logger.log(`inserted ${data.aid}`, "db-all_data");
	await client.queryObject(
		`INSERT INTO all_data (aid, bvid, description, uid, tags, title, published_at, duration)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (aid) DO NOTHING`,
		[
			data.aid,
			data.bvid,
			data.desc,
			data.owner.mid,
			null,
			data.title,
			formatTimestampToPsql(data.pubdate * SECOND + 8 * HOUR),
			data.duration,
		],
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

export async function getUnlabelledVideos(client: Client) {
	const queryResult = await client.queryObject<{ aid: number }>(
		`SELECT a.aid FROM all_data a LEFT JOIN labelling_result l ON a.aid = l.aid WHERE l.aid IS NULL`,
	);
	return queryResult.rows.map((row) => row.aid);
}

export async function insertVideoLabel(client: Client, aid: number, label: number) {
	return await client.queryObject(
		`INSERT INTO labelling_result (aid, label, model_version) VALUES ($1, $2, $3) ON CONFLICT (aid, model_version) DO NOTHING`,
		[aid, label, modelVersion],
	);
}

export async function getVideoInfoFromAllData(client: Client, aid: number) {
	const queryResult = await client.queryObject<AllDataType>(
		`SELECT * FROM all_data WHERE aid = $1`,
		[aid],
	);
	return queryResult.rows[0];
}
