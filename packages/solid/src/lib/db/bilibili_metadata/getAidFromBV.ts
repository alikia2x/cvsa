import { sql } from "@cvsa/core";

export async function getAidFromBV(bv: string) {
	const res = await sql`
        SELECT aid FROM bilibili_metadata WHERE bvid = ${bv}
    `;
	if (res.length <= 0) {
		return null;
	}
	const row = res[0];
	if (row && row.aid) {
		return Number(row.aid);
	}
	return null;
}
