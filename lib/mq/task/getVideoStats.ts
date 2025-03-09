import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getVideoInfo } from "lib/net/getVideoInfo.ts";

export async function insertVideoStats(client: Client, aid: number, task: string) {
	const data = await getVideoInfo(aid, task);
    const time = new Date().getTime();
    if (typeof data == 'number') {
		return data;
	}
    const views = data.stat.view;
    const danmakus = data.stat.danmaku;
    const replies = data.stat.reply;
    const likes = data.stat.like;
    const coins = data.stat.coin;
    const shares = data.stat.share;
    const favorites = data.stat.favorite;
    await client.queryObject(`
        INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `, [aid, views, danmakus, replies, likes, coins, shares, favorites]);
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
    }
}
