import { Job } from "bullmq";
import { insertVideoSnapshot } from "mq/task/getVideoStats";
import { sql } from "@core/db/dbNew";
import { lockManager } from "@core/mq/lockManager";

export const directSnapshotWorker = async (job: Job): Promise<void> => {
    const lock = await lockManager.isLocked(`directSnapshot-${job.data.aid}`);
    if (lock) {
        return;
    }
	const aid = job.data.aid;
	if (!aid) {
		throw new Error("aid does not exists");
	}
	await insertVideoSnapshot(sql, aid, "snapshotMilestoneVideo");
    await lockManager.acquireLock(`directSnapshot-${job.data.aid}`, 75);
};
