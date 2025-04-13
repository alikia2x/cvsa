import { Job } from "bullmq";
import { db } from "db/init.ts";
import {
	bulkGetVideosWithoutProcessingSchedules,
	bulkSetSnapshotStatus,
	getBulkSnapshotsInNextSecond,
	getSnapshotsInNextSecond,
	scheduleSnapshot,
	setSnapshotStatus,
	videoHasProcessingSchedule,
} from "db/snapshotSchedule.ts";
import { SECOND } from "@std/datetime";
import logger from "log/logger.ts";
import { SnapshotQueue } from "mq/index.ts";

const priorityMap: { [key: string]: number } = {
	"milestone": 1,
	"normal": 3,
};

export const bulkSnapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const schedules = await getBulkSnapshotsInNextSecond(client);
		const count = schedules.length;
		const groups = Math.ceil(count / 30);
		for (let i = 0; i < groups; i++) {
			const group = schedules.slice(i * 30, (i + 1) * 30);
			const aids = group.map((schedule) => Number(schedule.aid));
			const filteredAids = await bulkGetVideosWithoutProcessingSchedules(client, aids);
			if (filteredAids.length === 0) continue;
			await bulkSetSnapshotStatus(client, filteredAids, "processing");
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
	} finally {
		client.release();
	}
};

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const schedules = await getSnapshotsInNextSecond(client);
		for (const schedule of schedules) {
			if (await videoHasProcessingSchedule(client, Number(schedule.aid))) {
				continue;
			}
			let priority = 3;
			if (schedule.type && priorityMap[schedule.type]) {
				priority = priorityMap[schedule.type];
			}
			const aid = Number(schedule.aid);
			await setSnapshotStatus(client, schedule.id, "processing");
			await SnapshotQueue.add("snapshotVideo", {
				aid: Number(aid),
				id: Number(schedule.id),
				type: schedule.type ?? "normal",
			}, { priority });
		}
		return `OK`;
	} catch (e) {
		logger.error(e as Error);
	} finally {
		client.release();
	}
};

export const closetMilestone = (views: number) => {
	if (views < 100000) return 100000;
	if (views < 1000000) return 1000000;
	return 10000000;
};

export const scheduleCleanupWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const query = `
			SELECT id, aid, type 
			FROM snapshot_schedule
			WHERE status IN ('pending', 'processing')
				AND started_at < NOW() - INTERVAL '30 minutes'
		`;
		const { rows } = await client.queryObject<{ id: bigint; aid: bigint; type: string }>(query);
		if (rows.length === 0) return;
		for (const row of rows) {
			const id = Number(row.id);
			const aid = Number(row.aid);
			const type = row.type;
			await setSnapshotStatus(client, id, "timeout");
			await scheduleSnapshot(client, aid, type, Date.now() + 10 * SECOND);
			logger.log(
				`Schedule ${id} has no response received for 5 minutes, rescheduled.`,
				"mq",
				"fn:scheduleCleanupWorker",
			);
		}
	} catch (e) {
		logger.error(e as Error, "mq", "fn:scheduleCleanupWorker");
	} finally {
		client.release();
	}
};
