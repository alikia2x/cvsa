import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getVideoDetails } from "net/getVideoDetails.ts";
import { formatTimestampToPsql } from "utils/formatTimestampToPostgre.ts";
import logger from "log/logger.ts";
import { ClassifyVideoQueue } from "mq/index.ts";
import { userExistsInBiliUsers, videoExistsInAllData } from "db/allData.ts";
import { HOUR, SECOND } from "$std/datetime/constants.ts";

export async function insertVideoInfo(client: Client, aid: number) {
	const videoExists = await videoExistsInAllData(client, aid);
	if (videoExists) {
		return;
	}
	const data = await getVideoDetails(aid);
	if (data === null) {
		return null;
	}
	const bvid = data.View.bvid;
	const desc = data.View.desc;
	const uid = data.View.owner.mid;
	const tags = data.Tags
		.filter((tag) => !["old_channel", "topic"].indexOf(tag.tag_type))
		.map((tag) => tag.tag_name).join(",");
	const title = data.View.title;
	const published_at = formatTimestampToPsql(data.View.pubdate * SECOND + 8 * HOUR);
	const duration = data.View.duration;
	const cover = data.View.pic;
	await client.queryObject(
		`INSERT INTO bilibili_metadata (aid, bvid, description, uid, tags, title, published_at, duration, cover_url)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
		[aid, bvid, desc, uid, tags, title, published_at, duration, cover],
	);
	const userExists = await userExistsInBiliUsers(client, aid);
	if (!userExists) {
		await client.queryObject(
			`INSERT INTO bilibili_user (uid, username, "desc", fans) VALUES ($1, $2, $3, $4)`,
			[uid, data.View.owner.name, data.Card.card.sign, data.Card.follower],
		);
	} else {
		await client.queryObject(
			`UPDATE bilibili_user SET fans = $1 WHERE uid = $2`,
			[data.Card.follower, uid],
		);
	}

	const stat = data.View.stat;

	const query: string = `
				INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			`;
	await client.queryObject(
		query,
		[aid, stat.view, stat.danmaku, stat.reply, stat.like, stat.coin, stat.share, stat.favorite],
	);
	
	logger.log(`Inserted video metadata for aid: ${aid}`, "mq");
	await ClassifyVideoQueue.add("classifyVideo", { aid });
}
