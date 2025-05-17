import type { Psql } from "@core/db/global.d.ts";
import { parseTimestampFromPsql } from "utils/formatTimestampToPostgre.ts";

export async function getNotCollectedSongs(sql: Psql) {
    const rows = await sql<{ aid: number }[]>`
        SELECT lr.aid
        FROM labelling_result lr
        WHERE lr.label != 0
        AND NOT EXISTS (
            SELECT 1
            FROM songs s
            WHERE s.aid = lr.aid
        );
    `;
    return rows.map((row) => row.aid);
}

export async function aidExistsInSongs(sql: Psql, aid: number) {
    const rows = await sql<{ exists: boolean }[]>`
        SELECT EXISTS (
            SELECT 1
            FROM songs
            WHERE aid = ${aid}
        );
    `;
    return rows[0].exists;
}

export async function getSongsPublihsedAt(sql: Psql, aid: number) {
    const rows = await sql<{ published_at: string }[]>`
        SELECT published_at
        FROM songs
        WHERE aid = ${aid};
    `;
    if (rows.length === 0) {
        return null;
    }
    return parseTimestampFromPsql(rows[0].published_at);
}