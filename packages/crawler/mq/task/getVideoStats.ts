import { getVideoInfo } from "@core/net/getVideoInfo";
import logger from "@core/log";
import type { Psql } from "@core/db/psql.d";
import { insertVideoSnapshot } from "db/snapshot";

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
export async function takeVideoSnapshot(
	sql: Psql,
	aid: number,
	task: "snapshotMilestoneVideo" | "snapshotVideo"
): Promise<number | SnapshotNumber> {
	const r = await getVideoInfo(aid, task);
	if (typeof r == "number") {
		return r;
	}
	const { data, time } = r;
	const views = data.stat.view;
	const danmakus = data.stat.danmaku;
	const replies = data.stat.reply;
	const likes = data.stat.like;
	const coins = data.stat.coin;
	const shares = data.stat.share;
	const favorites = data.stat.favorite;

	await insertVideoSnapshot({
		createdAt: new Date(time).toISOString(),
		views,
		coins,
		likes,
		favorites,
		shares,
		danmakus,
		replies,
		aid
	});

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
