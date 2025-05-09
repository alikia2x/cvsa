import type { Psql } from "global.d.ts";
import { AllDataType, BiliUserType } from "@core/db/schema";
import { AkariModelVersion } from "ml/const";

export async function videoExistsInAllData(sql: Psql, aid: number) {
    const rows = await sql<{ exists: boolean }[]>`
        SELECT EXISTS(SELECT 1 FROM bilibili_metadata WHERE aid = ${aid})
    `;
    return rows[0].exists;
}

export async function userExistsInBiliUsers(sql: Psql, uid: number) {
    const rows = await sql<{ exists: boolean }[]>`
        SELECT EXISTS(SELECT 1 FROM bilibili_user WHERE uid = ${uid})
    `;
    return rows[0].exists;
}

export async function getUnlabelledVideos(sql: Psql) {
    const rows = await sql<{ aid: number }[]>`
        SELECT a.aid FROM bilibili_metadata a LEFT JOIN labelling_result l ON a.aid = l.aid WHERE l.aid IS NULL
    `;
    return rows.map((row) => row.aid);
}

export async function insertVideoLabel(sql: Psql, aid: number, label: number) {
    await sql`
        INSERT INTO labelling_result (aid, label, model_version) VALUES (${aid}, ${label}, ${AkariModelVersion}) ON CONFLICT (aid, model_version) DO NOTHING
    `;
}

export async function getVideoInfoFromAllData(sql: Psql, aid: number) {
    const rows = await sql<AllDataType[]>`
        SELECT * FROM bilibili_metadata WHERE aid = ${aid}
    `;
    const row = rows[0];
    let authorInfo = "";
    if (row.uid && await userExistsInBiliUsers(sql, row.uid)) {
        const userRows = await sql<BiliUserType[]>`
            SELECT * FROM bilibili_user WHERE uid = ${row.uid}
        `;
        const userRow = userRows[0];
        if (userRow) {
            authorInfo = userRow.desc;
        }
    }
    return {
        title: row.title,
        description: row.description,
        tags: row.tags,
        author_info: authorInfo,
    };
}

export async function getUnArchivedBiliUsers(sql: Psql) {
    const rows = await sql<{ uid: number }[]>`
        SELECT ad.uid
        FROM bilibili_metadata ad
        LEFT JOIN bilibili_user bu ON ad.uid = bu.uid
        WHERE bu.uid IS NULL;
    `;
    return rows.map((row) => row.uid);
}

export async function setBiliVideoStatus(sql: Psql, aid: number, status: number) {
    await sql`
        UPDATE bilibili_metadata SET status = ${status} WHERE aid = ${aid}
    `;
}

export async function getBiliVideoStatus(sql: Psql, aid: number) {
    const rows = await sql<{ status: number }[]>`
        SELECT status FROM bilibili_metadata WHERE aid = ${aid}
    `;
    if (rows.length === 0) return 0;
    return rows[0].status;
}