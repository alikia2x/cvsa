import { getVideoDetails } from "net/getVideoDetails";
import { formatTimestampToPsql } from "utils/formatTimestampToPostgre";
import logger from "@core/log";
import { ClassifyVideoQueue } from "mq/index";
import { userExistsInBiliUsers, videoExistsInAllData } from "db/bilibili_metadata";
import { HOUR, SECOND } from "@core/lib";
import type { Psql } from "@core/db/psql.d";
import { insertIntoSongs } from "./collectSongs";

export async function insertVideoInfo(sql: Psql, aid: number, insertSongs = false) {
	const videoExists = await videoExistsInAllData(sql, aid);
	if (videoExists && !insertSongs) {
		return;
	}
	if (videoExists && insertSongs) {
		await insertIntoSongs(sql, aid);
		return;
	}
	const data = await getVideoDetails(aid);
	if (data === null) {
		return null;
	}
	const bvid = data.View.bvid;
	const desc = data.View.desc;
	const uid = data.View.owner.mid;
	const tags = data.Tags.filter((tag) => !["old_channel", "topic"].indexOf(tag.tag_type))
		.map((tag) => tag.tag_name)
		.join(",");
	const title = data.View.title;
	const published_at = formatTimestampToPsql(data.View.pubdate * SECOND + 8 * HOUR);
	const duration = data.View.duration;
	const cover = data.View.pic;
	await sql`
		INSERT INTO bilibili_metadata (aid, bvid, description, uid, tags, title, published_at, duration, cover_url)
		VALUES (${aid}, ${bvid}, ${desc}, ${uid}, ${tags}, ${title}, ${published_at}, ${duration}, ${cover})
	`;
	const userExists = await userExistsInBiliUsers(sql, aid);
	if (!userExists) {
		await sql`
			INSERT INTO bilibili_user (uid, username, "desc", fans) 
			VALUES (${uid}, ${data.View.owner.name}, ${data.Card.card.sign}, ${data.Card.follower})
		`;
	} else {
		await sql`
			UPDATE bilibili_user SET fans = ${data.Card.follower} WHERE uid = ${uid}
		`;
	}

	const stat = data.View.stat;

	await sql`
		INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
		VALUES (
			${aid}, 
			${stat.view}, 
			${stat.danmaku}, 
			${stat.reply}, 
			${stat.like}, 
			${stat.coin}, 
			${stat.share}, 
			${stat.favorite}
		)
	`;

	logger.log(`Inserted video metadata for aid: ${aid}`, "mq");
	
	if (!insertSongs) {
		await ClassifyVideoQueue.add("classifyVideo", { aid });
		return;
	}
	await insertIntoSongs(sql, aid);
}
