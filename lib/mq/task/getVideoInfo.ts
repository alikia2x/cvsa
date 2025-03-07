import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getVideoInfo } from "lib/net/getVideoInfo.ts";
import { formatTimestampToPsql } from "lib/utils/formatTimestampToPostgre.ts";
import logger from "lib/log/logger.ts";
import { ClassifyVideoQueue } from "lib/mq/index.ts";
import { userExistsInBiliUsers, videoExistsInAllData } from "lib/db/allData.ts";

export async function insertVideoInfo(client: Client, aid: number) {
	const videoExists = await videoExistsInAllData(client, aid);
	if (videoExists) {
		return;
	}
	const data = await getVideoInfo(aid);
	if (data === null) {
		return null;
	}
	const bvid = data.View.bvid;
	const desc = data.View.desc;
	const uid = data.View.owner.mid;
	const tags = data.Tags
		.filter((tag) => tag.tag_type in ["old_channel", "topic"])
		.map((tag) => tag.tag_name).join(",");
	const title = data.View.title;
	const published_at = formatTimestampToPsql(data.View.pubdate);
	const duration = data.View.duration;
	await client.queryObject(
		`INSERT INTO all_data (aid, bvid, description, uid, tags, title, published_at, duration)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
		[aid, bvid, desc, uid, tags, title, published_at, duration],
	);
	const userExists = await userExistsInBiliUsers(client, aid);
	if (!userExists) {
		await client.queryObject(
			`INSERT INTO bili_user (uid, username, "desc", fans) VALUES ($1, $2, $3, $4)`,
			[uid, data.View.owner.name, data.Card.card.sign, data.Card.follower],
		);
	} else {
		await client.queryObject(
			`UPDATE bili_user SET fans = $1 WHERE uid = $2`,
			[data.Card.follower, uid],
		);
	}
	logger.log(`Inserted video metadata for aid: ${aid}`, "mq");
	await ClassifyVideoQueue.add("classifyVideo", { aid });
}
