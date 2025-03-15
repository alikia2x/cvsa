import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

export async function getUnsnapshotedSongs(client: Client) {
	const queryResult = await client.queryObject<{ aid: bigint }>(`
		SELECT DISTINCT s.aid
		FROM songs s
		LEFT JOIN video_snapshot v ON s.aid = v.aid
		WHERE v.aid IS NULL;
	`);
	return queryResult.rows.map((row) => Number(row.aid));
}
