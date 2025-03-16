import { Job } from "bullmq";
import { db } from "lib/db/init.ts";
import { getVideosNearMilestone } from "lib/db/snapshot.ts";
import { videoHasActiveSchedule } from "lib/db/snapshotSchedule.ts";

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		// TODO: implement
	} finally {
		client.release();
	}
};

export const collectMilestoneSnapshotsWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const videos = await getVideosNearMilestone(client);
		for (const video of videos) {
			if (await videoHasActiveSchedule(client, video.aid)) continue;
		}
	} catch (_e) {
		//
	} finally {
		client.release();
	}
};

export const takeSnapshotForVideoWorker = async (_job: Job) => {
	// TODO: implement
};
