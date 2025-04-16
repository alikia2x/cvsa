import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { AllDataType, BiliUserType } from "@core/db/schema";
import Akari from "ml/akari.ts";

export async function videoExistsInAllData(client: Client, aid: number) {
	return await client.queryObject<{ exists: boolean }>(
		`SELECT EXISTS(SELECT 1 FROM bilibili_metadata WHERE aid = $1)`,
		[aid],
	)
		.then((result) => result.rows[0].exists);
}

export async function userExistsInBiliUsers(client: Client, uid: number) {
	return await client.queryObject<{ exists: boolean }>(`SELECT EXISTS(SELECT 1 FROM bilibili_user WHERE uid = $1)`, [
		uid,
	]);
}

export async function getUnlabelledVideos(client: Client) {
	const queryResult = await client.queryObject<{ aid: number }>(
		`SELECT a.aid FROM bilibili_metadata a LEFT JOIN labelling_result l ON a.aid = l.aid WHERE l.aid IS NULL`,
	);
	return queryResult.rows.map((row) => row.aid);
}

export async function insertVideoLabel(client: Client, aid: number, label: number) {
	return await client.queryObject(
		`INSERT INTO labelling_result (aid, label, model_version) VALUES ($1, $2, $3) ON CONFLICT (aid, model_version) DO NOTHING`,
		[aid, label, Akari.getModelVersion()],
	);
}

export async function getVideoInfoFromAllData(client: Client, aid: number) {
	const queryResult = await client.queryObject<AllDataType>(
		`SELECT * FROM bilibili_metadata WHERE aid = $1`,
		[aid],
	);
	const row = queryResult.rows[0];
	let authorInfo = "";
	if (row.uid && await userExistsInBiliUsers(client, row.uid)) {
		const q = await client.queryObject<BiliUserType>(
			`SELECT * FROM bilibili_user WHERE uid = $1`,
			[row.uid],
		);
		const userRow = q.rows[0];
		if (userRow) {
			authorInfo = userRow.desc;
		}
	}
	return {
		title: row.title,
		description: row.description,
		tags: row.tags,
		author_info: authorInfo,
	};
}

export async function getUnArchivedBiliUsers(client: Client) {
	const queryResult = await client.queryObject<{ uid: number }>(
		`
		SELECT ad.uid
		FROM bilibili_metadata ad
		LEFT JOIN bilibili_user bu ON ad.uid = bu.uid
		WHERE bu.uid IS NULL;
		`,
		[],
	);
	const rows = queryResult.rows;
	return rows.map((row) => row.uid);
}

export async function setBiliVideoStatus(client: Client, aid: number, status: number) {
	return await client.queryObject(
		`UPDATE bilibili_metadata SET status = $1 WHERE aid = $2`,
		[status, aid],
	);
}

export async function getBiliVideoStatus(client: Client, aid: number) {
	const queryResult = await client.queryObject<{ status: number }>(
		`SELECT status FROM bilibili_metadata WHERE aid = $1`,
		[aid],
	);
	const rows = queryResult.rows;
	if (rows.length === 0) return 0;
	return rows[0].status;
}
