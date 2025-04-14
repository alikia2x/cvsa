import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

export async function getTimeoutSchedulesCount(client: Client) {
	const query: string = `
        SELECT COUNT(id)
        FROM snapshot_schedule
        WHERE status IN ('pending', 'processing')
          AND started_at < NOW() - INTERVAL '30 minutes'
	`;

	const { rows } = await client.queryObject<{ count: number }>(query);
	return rows[0].count;
}
