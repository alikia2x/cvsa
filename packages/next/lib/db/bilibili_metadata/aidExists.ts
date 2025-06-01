import { sql } from "@cvsa/core";

export async function aidExists(aid: number) {
	const res = await sql`
        SELECT 1 FROM bilibili_metadata WHERE aid = ${aid}
    `;
	return res.length > 0;
}
