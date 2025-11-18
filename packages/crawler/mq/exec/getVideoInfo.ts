import { Job } from "bullmq";
import { getVideoDetails } from "net/getVideoDetails";
import logger from "@core/log";
import { ClassifyVideoQueue, latestVideosEventsProducer } from "mq/index";
import {
	insertIntoMetadata,
	userExistsInBiliUsers,
	videoExistsInAllData
} from "db/bilibili_metadata";
import { insertIntoSongs } from "mq/task/collectSongs";
import { bilibiliUser, db, videoSnapshot } from "@core/drizzle";
import { eq } from "drizzle-orm";
import { GetVideoInfoJobData } from "mq/schema";

interface AddSongEventPayload {
	eventName: string;
	uid: string;
	songID: number;
}

const publishAddsongEvent = async (songID: number, uid: string) =>
	latestVideosEventsProducer.publishEvent<AddSongEventPayload>({
		eventName: "addSong",
		uid: uid,
		songID: songID
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
		description: data.View.desc,
		uid: uid,
		tags: data.Tags.filter((tag) => !["old_channel", "topic"].indexOf(tag.tag_type))
			.map((tag) => tag.tag_name)
			.join(","),
		title: data.View.title,
		publishedAt: new Date(data.View.pubdate * 1000).toISOString(),
		duration: data.View.duration,
		coverUrl: data.View.pic
	});

	const userExists = await userExistsInBiliUsers(aid);
	if (!userExists) {
		await db.insert(bilibiliUser).values({
			uid,
			username: data.View.owner.name,
			desc: data.Card.card.sign,
			fans: data.Card.follower
		});
	} else {
		await db
			.update(bilibiliUser)
			.set({ username: data.View.owner.name, desc: data.Card.card.sign })
			.where(eq(bilibiliUser.uid, uid));
	}

	const stat = data.View.stat;

	await db.insert(videoSnapshot).values({
		aid,
		views: stat.view,
		danmakus: stat.danmaku,
		replies: stat.reply,
		likes: stat.like,
		coins: stat.coin,
		shares: stat.share,
		favorites: stat.favorite
	});

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
