import { Job } from "bullmq";
import { db } from "db/init.ts";
import { getUnlabelledVideos, getVideoInfoFromAllData, insertVideoLabel } from "../../db/bilibili_metadata.ts";
import Akari from "ml/akari.ts";
import { ClassifyVideoQueue } from "mq/index.ts";
import logger from "log/logger.ts";
import { lockManager } from "mq/lockManager.ts";
import { aidExistsInSongs } from "db/songs.ts";
import { insertIntoSongs } from "mq/task/collectSongs.ts";
import { scheduleSnapshot } from "db/snapshotSchedule.ts";
import { MINUTE } from "@std/datetime";

export const classifyVideoWorker = async (job: Job) => {
	const client = await db.connect();
	const aid = job.data.aid;
	if (!aid) {
		return 3;
	}

	const videoInfo = await getVideoInfoFromAllData(client, aid);
	const title = videoInfo.title?.trim() || "untitled";
	const description = videoInfo.description?.trim() || "N/A";
	const tags = videoInfo.tags?.trim() || "empty";
	const label = await Akari.classifyVideo(title, description, tags, aid);
	if (label == -1) {
		logger.warn(`Failed to classify video ${aid}`, "ml");
	}
	await insertVideoLabel(client, aid, label);

	const exists = await aidExistsInSongs(client, aid);
	if (!exists && label !== 0) {
		await scheduleSnapshot(client, aid, "new", Date.now() + 10 * MINUTE, true);
		await insertIntoSongs(client, aid);
	}

	client.release();

	await job.updateData({
		...job.data,
		label: label,
	});

	return 0;
};

export const classifyVideosWorker = async () => {
	if (await lockManager.isLocked("classifyVideos")) {
		logger.log("job:classifyVideos is locked, skipping.", "mq");
		return;
	}

	await lockManager.acquireLock("classifyVideos");

	const client = await db.connect();
	const videos = await getUnlabelledVideos(client);
	logger.log(`Found ${videos.length} unlabelled videos`);
	client.release();

	let i = 0;
	for (const aid of videos) {
		if (i > 200) {
			await lockManager.releaseLock("classifyVideos");
			return 10000 + i;
		}
		await ClassifyVideoQueue.add("classifyVideo", { aid: Number(aid) });
		i++;
	}
	await lockManager.releaseLock("classifyVideos");
	return 0;
};
