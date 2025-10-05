import { getVideoInfo } from "@core/net/getVideoInfo";
import logger from "@core/log";
import type { Psql } from "@core/db/psql.d";

export interface SnapshotNumber {
	time: number;
	views: number;
	coins: number;
	likes: number;
	favorites: number;
	shares: number;
	danmakus: number;
	aid: number;
	replies: number;
}

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
export async function insertVideoSnapshot(sql: Psql, aid: number, task: string): Promise<number | SnapshotNumber> {
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

	await sql`
        INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
        VALUES (${aid}, ${views}, ${danmakus}, ${replies}, ${likes}, ${coins}, ${shares}, ${favorites})
    `;

	logger.log(`Taken snapshot for video ${aid}.`, "net", "fn:insertVideoSnapshot");

	return {
		aid,
		views,
		danmakus,
		replies,
		likes,
		coins,
		shares,
		favorites,
		time
	};
}
