import { Job } from "npm:bullmq@5.45.2";
import { db } from "db/init.ts";
import {
	bulkScheduleSnapshot,
	bulkSetSnapshotStatus,
	scheduleSnapshot,
	snapshotScheduleExists,
} from "db/snapshotSchedule.ts";
import { bulkGetVideoStats } from "net/bulkGetVideoStats.ts";
import logger from "log/logger.ts";
import { NetSchedulerError } from "@core/net/delegate.ts";
import { HOUR, MINUTE, SECOND } from "@std/datetime";
import { getRegularSnapshotInterval } from "../task/regularSnapshotInterval.ts";

export const takeBulkSnapshotForVideosWorker = async (job: Job) => {
	const dataMap: { [key: number]: number } = job.data.map;
	const ids = Object.keys(dataMap).map((id) => Number(id));
	const aidsToFetch: number[] = [];
	const client = await db.connect();
	try {
		for (const id of ids) {
			const aid = Number(dataMap[id]);
			const exists = await snapshotScheduleExists(client, id);
			if (!exists) {
				continue;
			}
			aidsToFetch.push(aid);
		}
		const data = await bulkGetVideoStats(aidsToFetch);
		if (typeof data === "number") {
			await bulkSetSnapshotStatus(client, ids, "failed");
			await bulkScheduleSnapshot(client, aidsToFetch, "normal", Date.now() + 15 * SECOND);
			return `GET_BILI_STATUS_${data}`;
		}
		for (const video of data) {
			const aid = video.id;
			const stat = video.cnt_info;
			const views = stat.play;
			const danmakus = stat.danmaku;
			const replies = stat.reply;
			const likes = stat.thumb_up;
			const coins = stat.coin;
			const shares = stat.share;
			const favorites = stat.collect;
			const query: string = `
				INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			`;
			await client.queryObject(
				query,
				[aid, views, danmakus, replies, likes, coins, shares, favorites],
			);

			logger.log(`Taken snapshot for video ${aid} in bulk.`, "net", "fn:takeBulkSnapshotForVideosWorker");
		}
		await bulkSetSnapshotStatus(client, ids, "completed");
		for (const aid of aidsToFetch) {
			const interval = await getRegularSnapshotInterval(client, aid);
			logger.log(`Scheduled regular snapshot for aid ${aid} in ${interval} hours.`, "mq");
			await scheduleSnapshot(client, aid, "normal", Date.now() + interval * HOUR);
		}
		return `DONE`;
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_PROXY_AVAILABLE") {
			logger.warn(
				`No available proxy for bulk request now.`,
				"mq",
				"fn:takeBulkSnapshotForVideosWorker",
			);
			await bulkSetSnapshotStatus(client, ids, "completed");
			await bulkScheduleSnapshot(client, aidsToFetch, "normal", Date.now() + 2 * MINUTE);
			return;
		}
		logger.error(e as Error, "mq", "fn:takeBulkSnapshotForVideosWorker");
		await bulkSetSnapshotStatus(client, ids, "failed");
	} finally {
		client.release();
	}
};
