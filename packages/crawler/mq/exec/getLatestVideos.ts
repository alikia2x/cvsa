import { sql } from "@core/db/dbNew";
import { Job } from "bullmq";
import { queueJobsCounter } from "metrics";
import { SnapshotQueue } from "mq";
import { queueLatestVideos } from "mq/task/queueLatestVideo";

export const getLatestVideosWorker = async (_job: Job): Promise<void> => {
	await queueLatestVideos(sql);
 	const counts = await SnapshotQueue.getJobCounts();
	const waiting = counts?.waiting;
	const prioritized = counts?.prioritized;
	waiting && queueJobsCounter.record(waiting, { queueName: "SnapshotQueue", status: "waiting" });
	prioritized && queueJobsCounter.record(prioritized, { queueName: "SnapshotQueue", status: "prioritized" });
};
