import { Job } from "bullmq";
import { getUnlabelledVideos, getVideoInfoFromAllData, insertVideoLabel } from "../../db/bilibili_metadata.ts";
import Akari from "ml/akari.ts";
import { ClassifyVideoQueue } from "mq/index.ts";
import logger from "@core/log/logger.ts";
import { lockManager } from "@core/mq/lockManager.ts";
import { aidExistsInSongs } from "db/songs.ts";
import { insertIntoSongs } from "mq/task/collectSongs.ts";
import { scheduleSnapshot } from "db/snapshotSchedule.ts";
import { MINUTE } from "@core/const/time.ts";
import { sql } from "@core/db/dbNew.ts";

export const classifyVideoWorker = async (job: Job) => {
	const aid = job.data.aid;
	if (!aid) {
		return 3;
	}

	const videoInfo = await getVideoInfoFromAllData(sql, aid);
	const title = videoInfo.title?.trim() || "untitled";
	const description = videoInfo.description?.trim() || "N/A";
	const tags = videoInfo.tags?.trim() || "empty";
	const label = await Akari.classifyVideo(title, description, tags, aid);
	if (label == -1) {
		logger.warn(`Failed to classify video ${aid}`, "ml");
	}
	await insertVideoLabel(sql, aid, label);

	const exists = await aidExistsInSongs(sql, aid);
	if (!exists && label !== 0) {
		await scheduleSnapshot(sql, aid, "new", Date.now() + 10 * MINUTE, true);
		await insertIntoSongs(sql, aid);
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

	const videos = await getUnlabelledVideos(sql);
	logger.log(`Found ${videos.length} unlabelled videos`);

	const startTime = new Date().getTime();
	for (const aid of videos) {
		const now = new Date().getTime();
		if (now - startTime > 4.2 * MINUTE) {
			await lockManager.releaseLock("classifyVideos");
			return 1;
		}
		await ClassifyVideoQueue.add("classifyVideo", { aid: Number(aid) });
	}
	await lockManager.releaseLock("classifyVideos");
	return 0;
};
