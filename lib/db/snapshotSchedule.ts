import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

/* 
    Returns true if the specified `aid` has at least one record with "pending" or "processing" status.
*/
export async function videoHasActiveSchedule(client: Client, aid: number) {
    const res = await client.queryObject<{ status: string }>(
        `SELECT status FROM snapshot_schedule WHERE aid = $1 AND (status = 'pending' OR status = 'processing')`,
        [aid],
    );
    return res.rows.length > 0;
}