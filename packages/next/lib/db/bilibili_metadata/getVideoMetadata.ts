import { BiliVideoMetadataType, sql } from "@cvsa/core";

export async function getVideoMetadata(aid: number) {
	const res = await sql<BiliVideoMetadataType[]>`
        SELECT * FROM bilibili_metadata WHERE aid = ${aid}
    `;
	if (res.length <= 0) {
		return null;
	}
	const row = res[0];
	if (row) {
		return row;
	}
	return null;
}
