import sql from "./db";
import type { VideoSnapshotType } from "@core/db/schema.d.ts";

export async function getVideoSnapshots(
	aid: number,
	limit: number,
	pageOrOffset: number,
	reverse: boolean,
	mode: "page" | "offset" = "page"
) {
	const offset = mode === "page" ? (pageOrOffset - 1) * limit : pageOrOffset;
	if (reverse) {
		return sql<VideoSnapshotType[]>`
            SELECT *
            FROM video_snapshot
            WHERE aid = ${aid}
            ORDER BY created_at
            LIMIT ${limit} OFFSET ${offset}
		`;
	} else {
		return sql<VideoSnapshotType[]>`
            SELECT *
            FROM video_snapshot
            WHERE aid = ${aid}
            ORDER BY created_at DESC
            LIMIT ${limit} OFFSET ${offset}
		`;
	}
}

export async function getVideoSnapshotsByBV(
	bv: string,
	limit: number,
	pageOrOffset: number,
	reverse: boolean,
	mode: "page" | "offset" = "page"
) {
	const offset = mode === "page" ? (pageOrOffset - 1) * limit : pageOrOffset;
	if (reverse) {
		return sql<VideoSnapshotType[]>`
            SELECT vs.*
            FROM video_snapshot vs
                     JOIN bilibili_metadata bm ON vs.aid = bm.aid
            WHERE bm.bvid = ${bv}
            ORDER BY vs.created_at
            LIMIT ${limit} OFFSET ${offset}
		`;
	} else {
		return sql<VideoSnapshotType[]>`
            SELECT vs.*
            FROM video_snapshot vs
                     JOIN bilibili_metadata bm ON vs.aid = bm.aid
            WHERE bm.bvid = ${bv}
            ORDER BY vs.created_at DESC
            LIMIT ${limit} OFFSET ${offset}
		`;
	}
}
