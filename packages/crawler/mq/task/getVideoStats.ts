import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getVideoInfo } from "net/getVideoInfo.ts";
import { LatestSnapshotType } from "@core/db/schema.d.ts";
import logger from "log/logger.ts";

/*
 * Fetch video stats from bilibili API and insert into database
 * @returns {Promise<number|VideoSnapshot>}
 * A number indicating the status code when receiving non-0 status code from bilibili,
 * otherwise an VideoSnapshot object containing the video stats
 * @throws {NetSchedulerError} - The error will be thrown in following cases:
 * - No proxy is available currently: with error code `NO_PROXY_AVAILABLE`
 * - The native `fetch` function threw an error: with error code `FETCH_ERROR`
 * - The alicloud-fc threw an error: with error code `ALICLOUD_FC_ERROR`
 */
export async function insertVideoSnapshot(
	client: Client,
	aid: number,
	task: string,
): Promise<number | LatestSnapshotType> {
	const data = await getVideoInfo(aid, task);
	if (typeof data == "number") {
		return data;
	}
	const time = new Date().getTime();
	const views = data.stat.view;
	const danmakus = data.stat.danmaku;
	const replies = data.stat.reply;
	const likes = data.stat.like;
	const coins = data.stat.coin;
	const shares = data.stat.share;
	const favorites = data.stat.favorite;

	const query: string = `
        INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `;
	await client.queryObject(
		query,
		[aid, views, danmakus, replies, likes, coins, shares, favorites],
	);

	logger.log(`Taken snapshot for video ${aid}.`, "net", "fn:insertVideoSnapshot");

	const snapshot: LatestSnapshotType = {
		aid,
		views,
		danmakus,
		replies,
		likes,
		coins,
		shares,
		favorites,
		time,
	};

	return snapshot;
}
