import { DAY, HOUR, MINUTE, SECOND } from "$std/datetime/constants.ts";
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

interface Snapshot {
    created_at: Date;
    views: number;
}

export async function findClosestSnapshot(
    client: Client,
    aid: number,
    targetTime: Date
): Promise<Snapshot | null> {
    const query = `
        SELECT created_at, views FROM video_snapshot
        WHERE aid = $1
        ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - $2::timestamptz))) ASC
        LIMIT 1
    `;
    const result = await client.queryObject<{ created_at: string; views: number }>(
        query,
        [aid, targetTime.toISOString()]
    );
    if (result.rows.length === 0) return null;
    const row = result.rows[0];
    return {
        created_at: new Date(row.created_at),
        views: row.views,
    };
}

export async function getShortTermTimeFeaturesForVideo(
    client: Client,
    aid: number,
    initialTimestampMiliseconds: number
): Promise<number[]> {
    const initialTime = new Date(initialTimestampMiliseconds);
    const timeWindows = [
        [ 5 * MINUTE,  0 * MINUTE],
        [ 15 * MINUTE,  0 * MINUTE],
        [ 40 * MINUTE,  0 * MINUTE],
        [ 1 * HOUR,  0 * HOUR],
        [ 2 * HOUR,  1 * HOUR],
        [ 3 * HOUR,  2 * HOUR],
        [ 3 * HOUR,  0 * HOUR],
        [ 6 * HOUR,  0 * HOUR],
        [18 * HOUR, 12 * HOUR],
        [ 1 * DAY,   0 * DAY],
        [ 3 * DAY,   0 * DAY],
        [ 7 * DAY,   0 * DAY]
    ];

    const results: number[] = [];

    for (const [windowStart, windowEnd] of timeWindows) {
        const targetTimeStart = new Date(initialTime.getTime() - windowStart);
        const targetTimeEnd = new Date(initialTime.getTime() - windowEnd);

        const startRecord = await findClosestSnapshot(client, aid, targetTimeStart);
        const endRecord = await findClosestSnapshot(client, aid, targetTimeEnd);

        if (!startRecord || !endRecord) {
            results.push(NaN);
            continue;
        }

        const timeDiffSeconds = 
            (endRecord.created_at.getTime() - startRecord.created_at.getTime()) / 1000;
        const windowDuration = windowStart - windowEnd;

        let scale = 0;
        if (windowDuration > 0) {
            scale = timeDiffSeconds / windowDuration;
        }

        const viewsDiff = endRecord.views - startRecord.views;
        const adjustedViews = Math.max(viewsDiff, 1);

        let result: number;
        if (scale > 0) {
            result = Math.log2(adjustedViews / scale + 1);
        } else {
            result = Math.log2(adjustedViews + 1);
        }

        results.push(result);
    }

    return results;
}