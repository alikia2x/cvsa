import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getUnlabelledVideos, getVideoInfoFromAllData, insertVideoLabel } from "lib/db/allData.ts";
import { classifyVideo } from "lib/ml/filter_inference.ts";
import { ClassifyVideoQueue } from "lib/mq/index.ts";
import logger from "lib/log/logger.ts";
import { lockManager } from "lib/mq/lockManager.ts";

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
	const label = await classifyVideo(title, description, tags, aid);
	if (label == -1) {
		logger.warn(`Failed to classify video ${aid}`, "ml");
	}
	await insertVideoLabel(client, aid, label);

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
