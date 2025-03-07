import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

export async function getNotCollectedSongs(client: Client) {
	const queryResult = await client.queryObject<{ aid: number }>(`
		SELECT lr.aid
		FROM labelling_result lr
		WHERE lr.label != 0
		AND NOT EXISTS (
			SELECT 1
			FROM songs s
			WHERE s.aid = lr.aid
		);
	`);
	return queryResult.rows.map((row) => row.aid);
}

export async function aidExistsInSongs(client: Client, aid: number) {
	const queryResult = await client.queryObject<{ exists: boolean }>(
		`
	    SELECT EXISTS (
	        SELECT 1
	        FROM songs
	        WHERE aid = $1
	    );
	`,
		[aid],
	);
	return queryResult.rows[0].exists;
}
