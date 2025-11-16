import { queueJobsCounter } from "metrics";
import { SnapshotQueue } from "mq";

export const collectQueueMetrics = async () => {
	const counts = await SnapshotQueue.getJobCounts();
	const waiting = counts?.waiting;
	const prioritized = counts?.prioritized;
	const active = counts?.active;
	const completed = counts?.completed;
	const failed = counts?.failed;
	const delayed = counts?.delayed;
	waiting && queueJobsCounter.record(waiting, { queueName: "SnapshotQueue", status: "waiting" });
	prioritized &&
		queueJobsCounter.record(prioritized, { queueName: "SnapshotQueue", status: "prioritized" });
	active && queueJobsCounter.record(active, { queueName: "SnapshotQueue", status: "active" });
	completed &&
		queueJobsCounter.record(completed, { queueName: "SnapshotQueue", status: "completed" });
	failed && queueJobsCounter.record(failed, { queueName: "SnapshotQueue", status: "failed" });
	delayed && queueJobsCounter.record(delayed, { queueName: "SnapshotQueue", status: "delayed" });
};
