import {Client} from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import {AllDataType, BiliUserType} from "lib/db/schema.d.ts";
import {modelVersion} from "lib/ml/filter_inference.ts";

export async function videoExistsInAllData(client: Client, aid: number) {
	return await client.queryObject<{ exists: boolean }>(`SELECT EXISTS(SELECT 1 FROM all_data WHERE aid = $1)`, [aid])
		.then((result) => result.rows[0].exists);
}

export async function userExistsInBiliUsers(client: Client, uid: number) {
	return await client.queryObject<{ exists: boolean }>(`SELECT EXISTS(SELECT 1 FROM bili_user WHERE uid = $1)`, [uid])
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
	const row = queryResult.rows[0];
	let authorInfo = "";
	if (row.uid && await userExistsInBiliUsers(client, row.uid)) {
		const q = await client.queryObject<BiliUserType>(
			`SELECT * FROM bili_user WHERE uid = $1`,
			[row.uid],
		)
		const userRow = q.rows[0];
		authorInfo = userRow.desc;
	}
	return {
		title: row.title,
		description: row.description,
		tags: row.tags,
		author_info: authorInfo
	};
}
