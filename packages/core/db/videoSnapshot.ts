import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { VideoSnapshotType } from "./schema.d.ts";

export async function getVideoSnapshots(client: Client, aid: number, limit: number, pageOrOffset: number, reverse: boolean, mode: 'page' | 'offset' = 'page') {
    const offset = mode === 'page' ? (pageOrOffset - 1) * limit : pageOrOffset;
    const order = reverse ? 'ASC' : 'DESC';
    const query = `
        SELECT *
        FROM video_snapshot
        WHERE aid = $1
        ORDER BY created_at ${order}
        LIMIT $2
        OFFSET $3
    `;
    const queryResult = await client.queryObject<VideoSnapshotType>(query, [aid, limit, offset]);
    return queryResult.rows;
}

export async function getVideoSnapshotsByBV(client: Client, bv: string, limit: number, pageOrOffset: number, reverse: boolean, mode: 'page' | 'offset' = 'page') {
    const offset = mode === 'page' ? (pageOrOffset - 1) * limit : pageOrOffset;
    const order = reverse ? 'ASC' : 'DESC';
    const query = `
        SELECT vs.*
        FROM video_snapshot vs
        JOIN bilibili_metadata bm ON vs.aid = bm.aid
        WHERE bm.bvid = $1
        ORDER BY vs.created_at ${order}
        LIMIT $2
        OFFSET $3
    `
    const queryResult = await client.queryObject<VideoSnapshotType>(query, [bv, limit, offset]);
    return queryResult.rows;
}