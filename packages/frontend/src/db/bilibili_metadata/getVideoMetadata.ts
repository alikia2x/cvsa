import { sql } from "@core/db/dbNew";

export async function getVideoMetadata(aid: number) {
	const res = await sql`
        SELECT * FROM bilibili_metadata WHERE aid = ${aid}
    `;
	if (res.length <= 0) {
		return null;
	}
	const row = res[0];
	if (row) {
		return row;
	}
	return {};
}
