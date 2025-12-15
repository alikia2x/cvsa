import logger from "@core/log";
import type { Job } from "bullmq";
import { removeAllTimeoutSchedules } from "mq/task/removeAllTimeoutSchedules";

export const scheduleCleanupWorker = async (_job: Job): Promise<void> => {
	try {
		const row = await removeAllTimeoutSchedules();
		if (row.length > 0 && row[0].deleted) {
			logger.log(
				`Removed ${row[0].deleted} timeout schedules.`,
				"mq",
				"fn:scheduleCleanupWorker"
			);
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:scheduleCleanupWorker");
	}
};
