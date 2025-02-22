import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getUnlabelledVideos, getVideoInfoFromAllData, insertVideoLabel} from "lib/db/allData.ts";
import { classifyVideo, initializeModels } from "lib/ml/filter_inference.ts";
import { ClassifyVideoQueue } from "lib/mq/index.ts";
import logger from "lib/log/logger.ts";

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
	const authorInfo = "No";
	const label = await classifyVideo(title, description, tags, authorInfo, aid);
	if (label == -1) {
		logger.warn(`Failed to classify video ${aid}`, "ml");
	}
	insertVideoLabel(client, aid, label);

	client.release();

	job.updateData({
		...job.data, label: label,
	});

	return 0;
};

export const classifyVideosWorker = async () => {
	await initializeModels();
	const client = await db.connect();
	const videos = await getUnlabelledVideos(client);
	logger.log(`Found ${videos.length} unlabelled videos`)
	client.release();
	let i = 0;
	for (const aid of videos) {
		if (i > 200) return 10000 + i;
		await ClassifyVideoQueue.add("classifyVideo", { aid: Number(aid) });
		i++;
	}
	return 0;
};
