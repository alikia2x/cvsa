import { Job } from "bullmq";
import {
	getUnlabelledVideos,
	getVideoInfoFromAllData,
	insertVideoLabel
} from "../../db/bilibili_metadata";
import Akari from "ml/akari_api";
import { ClassifyVideoQueue } from "mq/index";
import logger from "@core/log";
import { lockManager } from "@core/mq/lockManager";
import { aidExistsInSongs } from "db/songs";
import { insertIntoSongs } from "mq/task/collectSongs";
import { scheduleSnapshot } from "db/snapshotSchedule";
import { MINUTE } from "@core/lib";
import { sql } from "@core/db/dbNew";

export const classifyVideoWorker = async (job: Job) => {
	const aid = job.data.aid;
	if (!aid) {
		return 3;
	}

	const videoInfo = await getVideoInfoFromAllData(aid);
	if (!videoInfo) {
		return 3;
	}
	const title = videoInfo.title.trim();
	const description = videoInfo.description.trim();
	const tags = videoInfo.tags?.trim() || "empty";
	const label = await Akari.classifyVideo(title, description, tags, aid);
	if (label == -1) {
		logger.warn(`Failed to classify video ${aid}`, "ml");
	}
	await insertVideoLabel(aid, label);

	const exists = await aidExistsInSongs(sql, aid);
	if (!exists && label !== 0) {
		await scheduleSnapshot(sql, aid, "new", Date.now() + 1.5 * MINUTE, true);
		await scheduleSnapshot(sql, aid, "new", Date.now() + 3 * MINUTE, true);
		await scheduleSnapshot(sql, aid, "new", Date.now() + 5 * MINUTE, true);
		await scheduleSnapshot(sql, aid, "new", Date.now() + 10 * MINUTE, true);
		await insertIntoSongs(aid);
	}

	await job.updateData({
		...job.data,
		label: label
	});

	return 0;
};

export const classifyVideosWorker = async () => {
	if (await lockManager.isLocked("classifyVideos")) {
		logger.log("job:classifyVideos is locked, skipping.", "mq");
		return;
	}

	await lockManager.acquireLock("classifyVideos", 5 * 60);

	const videos = await getUnlabelledVideos();
	logger.log(`Found ${videos.length} unlabelled videos`);

	const startTime = Date.now();
	for (const aid of videos) {
		const now = Date.now();
		if (now - startTime > 4.2 * MINUTE) {
			await lockManager.releaseLock("classifyVideos");
			return 1;
		}
		await ClassifyVideoQueue.add("classifyVideo", { aid: Number(aid) });
	}
	await lockManager.releaseLock("classifyVideos");
	return 0;
};
