import { bilibiliUser, db, videoSnapshot } from "@core/drizzle";
import logger from "@core/log";
import { getVideoDetails } from "@core/net/getVideoDetails";
import type { Job } from "bullmq";
import {
	insertIntoMetadata,
	userExistsInBiliUsers,
	videoExistsInAllData,
} from "db/bilibili_metadata";
import { eq } from "drizzle-orm";
import { snapshotCounter } from "metrics";
import { ClassifyVideoQueue, latestVideosEventsProducer } from "mq/index";
import type { GetVideoInfoJobData } from "mq/schema";
import { insertIntoSongs } from "mq/task/collectSongs";

interface AddSongEventPayload {
	eventName: string;
	uid: string;
	songID: number;
}

const publishAddsongEvent = async (songID: number, uid: string) =>
	latestVideosEventsProducer.publishEvent<AddSongEventPayload>({
		eventName: "addSong",
		songID: songID,
		uid: uid,
	});

export const getVideoInfoWorker = async (job: Job<GetVideoInfoJobData>): Promise<void> => {
	const aid = job.data.aid;
	const insertSongs = job.data.insertSongs || false;
	if (!aid) {
		logger.warn("aid does not exists", "mq", "job:getVideoInfo");
		return;
	}
	const videoExists = await videoExistsInAllData(aid);
	if (videoExists && !insertSongs) {
		return;
	}
	if (videoExists && insertSongs) {
		const songs = await insertIntoSongs(aid);
		if (songs.length === 0) {
			logger.warn(`Failed to insert song for aid: ${aid}`, "mq", "fn:getVideoInfoWorker");
			return;
		}
		await publishAddsongEvent(songs[0].id, job.data.uid);
		return;
	}
	const data = await getVideoDetails(aid);
	if (data === null) {
		return null;
	}

	const uid = data.View.owner.mid;

	await insertIntoMetadata({
		aid,
		bvid: data.View.bvid,
		coverUrl: data.View.pic,
		description: data.View.desc,
		duration: data.View.duration,
		publishedAt: new Date(data.View.pubdate * 1000).toISOString(),
		tags: data.Tags.filter((tag) => !["old_channel", "topic"].indexOf(tag.tag_type))
			.map((tag) => tag.tag_name)
			.join(","),
		title: data.View.title,
		uid: uid,
	});

	const userExists = await userExistsInBiliUsers(aid);
	if (!userExists) {
		await db.insert(bilibiliUser).values({
			avatar: data.View.owner.face,
			desc: data.Card.card.sign,
			fans: data.Card.follower,
			uid,
			username: data.View.owner.name,
		});
	} else {
		await db
			.update(bilibiliUser)
			.set({
				avatar: data.View.owner.face,
				desc: data.Card.card.sign,
				fans: data.Card.follower,
				username: data.View.owner.name,
			})
			.where(eq(bilibiliUser.uid, uid));
	}

	const stat = data.View.stat;

	await db.insert(videoSnapshot).values({
		aid,
		coins: stat.coin,
		danmakus: stat.danmaku,
		favorites: stat.favorite,
		likes: stat.like,
		replies: stat.reply,
		shares: stat.share,
		views: stat.view,
	});

	snapshotCounter.add(1);

	logger.log(`Inserted video metadata for aid: ${aid}`, "mq");

	if (!insertSongs) {
		await ClassifyVideoQueue.add("classifyVideo", { aid });
		return;
	}
	const songs = await insertIntoSongs(aid);
	if (songs.length === 0) {
		logger.warn(`Failed to insert song for aid: ${aid}`, "mq", "fn:getVideoInfoWorker");
		return;
	}
	await publishAddsongEvent(songs[0].id, job.data.uid);
};
