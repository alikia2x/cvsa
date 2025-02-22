import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getUnlabeledVideos, getVideoInfoFromAllData, insertVideoLabel} from "lib/db/allData.ts";
import { classifyVideo, initializeModels } from "lib/ml/filter_inference.ts";
import { ClassifyVideoQueue } from "lib/mq/index.ts";

export const classifyVideoWorker = async (job: Job) => {
	const client = await db.connect();
	const aid = job.data.aid;
	if (!aid) {
		return 3;
	}

	const videoInfo = await getVideoInfoFromAllData(client, aid);
	const label = await classifyVideo(videoInfo.title ?? "", videoInfo.description ?? "", videoInfo.tags ?? "", "", aid);
	insertVideoLabel(client, aid, label);
	
	client.release();
	return 0;
};

export const classifyVideosWorker = async () => {
	await initializeModels();
	const client = await db.connect();
	const videos = await getUnlabeledVideos(client);
	client.release();
	for (const aid of videos) {
		await ClassifyVideoQueue.add("classifyVideo", { aid });
	}
};
