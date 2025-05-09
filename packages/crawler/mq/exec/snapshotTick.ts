import { Job } from "bullmq";
import {
	bulkGetVideosWithoutProcessingSchedules,
	bulkSetSnapshotStatus,
	getBulkSnapshotsInNextSecond,
	getSnapshotsInNextSecond,
	setSnapshotStatus,
	videoHasProcessingSchedule,
} from "db/snapshotSchedule.ts";
import logger from "@core/log/logger.ts";
import { SnapshotQueue } from "mq/index.ts";
import { sql } from "@core/db/dbNew";

const priorityMap: { [key: string]: number } = {
	"milestone": 1,
	"normal": 3,
};

export const bulkSnapshotTickWorker = async (_job: Job) => {
	try {
		const schedules = await getBulkSnapshotsInNextSecond(sql);
		const count = schedules.length;
		const groups = Math.ceil(count / 30);
		for (let i = 0; i < groups; i++) {
			const group = schedules.slice(i * 30, (i + 1) * 30);
			const aids = group.map((schedule) => Number(schedule.aid));
			const filteredAids = await bulkGetVideosWithoutProcessingSchedules(sql, aids);
			if (filteredAids.length === 0) continue;
			await bulkSetSnapshotStatus(sql, filteredAids, "processing");
			const schedulesData = group.map((schedule) => {
				return {
					aid: Number(schedule.aid),
					id: Number(schedule.id),
					type: schedule.type,
					created_at: schedule.created_at,
					started_at: schedule.started_at,
					finished_at: schedule.finished_at,
					status: schedule.status,
				};
			});
			await SnapshotQueue.add("bulkSnapshotVideo", {
				schedules: schedulesData,
			}, { priority: 3 });
		}
		return `OK`;
	} catch (e) {
		logger.error(e as Error);
	}
};

export const snapshotTickWorker = async (_job: Job) => {
	try {
		const schedules = await getSnapshotsInNextSecond(sql);
		for (const schedule of schedules) {
			if (await videoHasProcessingSchedule(sql, Number(schedule.aid))) {
				continue;
			}
			let priority = 3;
			if (schedule.type && priorityMap[schedule.type]) {
				priority = priorityMap[schedule.type];
			}
			const aid = Number(schedule.aid);
			await setSnapshotStatus(sql, schedule.id, "processing");
			await SnapshotQueue.add("snapshotVideo", {
				aid: Number(aid),
				id: Number(schedule.id),
				type: schedule.type ?? "normal",
			}, { priority });
		}
		return `OK`;
	} catch (e) {
		logger.error(e as Error);
	}
};

export const closetMilestone = (views: number) => {
	if (views < 100000) return 100000;
	if (views < 1000000) return 1000000;
	return 10000000;
};
