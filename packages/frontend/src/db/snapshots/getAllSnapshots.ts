import { sql } from "@core/db/dbNew";

export async function getAllSnapshots(aid: number) {
	const res = await sql`
        SELECT * FROM video_snapshot WHERE aid = ${aid} ORDER BY created_at DESC
    `;
	if (res.length <= 0) {
		return null;
	}
	return res;
}
