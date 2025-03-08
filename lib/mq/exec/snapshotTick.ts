import { Job } from "bullmq";
import { MINUTE, SECOND } from "$std/datetime/constants.ts";
import { db } from "lib/db/init.ts";
import { getSongsNearMilestone, getUnsnapshotedSongs } from "lib/db/snapshot.ts";
import { SnapshotQueue } from "lib/mq/index.ts";
import { insertVideoStats } from "lib/mq/task/getVideoStats.ts";
import { parseTimestampFromPsql } from "lib/utils/formatTimestampToPostgre.ts";
import { redis } from "lib/db/redis.ts";
import { NetSchedulerError } from "lib/mq/scheduler.ts";
import logger from "lib/log/logger.ts";

async function snapshotScheduled(aid: number) {
	try {
		return await redis.exists(`cvsa:snapshot:${aid}`);
	} catch {
		logger.error(`Failed to check scheduled status for ${aid}`, "mq");
		return false;
	}
}

async function setSnapshotScheduled(aid: number, value: boolean, exp: number) {
	try {
		if (value) {
			await redis.set(`cvsa:snapshot:${aid}`, 1, "EX", exp);
		} else {
			await redis.del(`cvsa:snapshot:${aid}`);
		}
	} catch {
		logger.error(`Failed to set scheduled status to ${value} for ${aid}`, "mq");
	}
}

interface SongNearMilestone {
	aid: number;
	id: number;
	created_at: string;
	views: number;
	coins: number;
	likes: number;
	favorites: number;
	shares: number;
	danmakus: number;
	replies: number;
}

async function processMilestoneSnapshots(vidoesNearMilestone: SongNearMilestone[]) {
	let i = 0;
	for (const snapshot of vidoesNearMilestone) {
		if (await snapshotScheduled(snapshot.aid)) {
			continue;
		}
		const factor = Math.floor(i / 8);
		const delayTime = factor * SECOND * 2;
		SnapshotQueue.add("snapshotMilestoneVideo", {
			aid: snapshot.aid,
			currentViews: snapshot.views,
			snapshotedAt: snapshot.created_at,
		}, { delay: delayTime });
		await setSnapshotScheduled(snapshot.aid, true, 20 * 60);
		i++;
	}
}

async function processUnsnapshotedVideos(unsnapshotedVideos: number[]) {
	let i = 0;
	for (const aid of unsnapshotedVideos) {
		if (await snapshotScheduled(aid)) {
			continue;
		}
		const factor = Math.floor(i / 5);
		const delayTime = factor * SECOND * 4;
		SnapshotQueue.add("snapshotVideo", {
			aid,
		}, { delay: delayTime });
		await setSnapshotScheduled(aid, true, 6 * 60 * 60);
		i++;
	}
}

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const vidoesNearMilestone = await getSongsNearMilestone(client);
		await processMilestoneSnapshots(vidoesNearMilestone);

		const unsnapshotedVideos = await getUnsnapshotedSongs(client);
		await processUnsnapshotedVideos(unsnapshotedVideos);
	} finally {
		client.release();
	}
};

export const takeSnapshotForMilestoneVideoWorker = async (job: Job) => {
	const client = await db.connect();
	await setSnapshotScheduled(job.data.aid, true, 20 * 60);
	try {
		const { aid, currentViews, lastSnapshoted } = job.data;
		const stat = await insertVideoStats(client, aid, "snapshotMilestoneVideo");
		if (stat == null) {
			setSnapshotScheduled(aid, false, 0);
			return;
		}
		const nextMilestone = currentViews >= 100000 ? 1000000 : 100000;
		if (stat.views >= nextMilestone) {
			setSnapshotScheduled(aid, false, 0);
			return;
		}
		const intervalSeconds = (Date.now() - parseTimestampFromPsql(lastSnapshoted)) / SECOND;
		const viewsIncrement = stat.views - currentViews;
		const incrementSpeed = viewsIncrement / intervalSeconds;
		const viewsToIncrease = nextMilestone - stat.views;
		const eta = viewsToIncrease / incrementSpeed;
		const scheduledNextSnapshotDelay = eta * SECOND / 3;
		const maxInterval = 20 * MINUTE;
		const delay = Math.min(scheduledNextSnapshotDelay, maxInterval);
		SnapshotQueue.add("snapshotMilestoneVideo", {
			aid,
			currentViews: stat.views,
			snapshotedAt: stat.time,
		}, { delay });
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_AVAILABLE_PROXY") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:takeSnapshotForMilestoneVideoWorker",
			);
			SnapshotQueue.add("snapshotMilestoneVideo", {
				aid: job.data.aid,
				currentViews: job.data.currentViews,
				snapshotedAt: job.data.snapshotedAt,
			}, { delay: 5 * SECOND });
			return;
		}
		throw e;
	} finally {
		client.release();
	}
};

export const takeSnapshotForVideoWorker = async (job: Job) => {
	const client = await db.connect();
	await setSnapshotScheduled(job.data.aid, true, 6 * 60 * 60);
	try {
		const { aid } = job.data;
		const stat = await insertVideoStats(client, aid, "getVideoInfo");
		if (stat == null) {
			setSnapshotScheduled(aid, false, 0);
			return;
		}
		const nearMilestone = (stat.views >= 90000 && stat.views < 100000) ||
			(stat.views >= 900000 && stat.views < 1000000);
		if (nearMilestone) {
			SnapshotQueue.add("snapshotMilestoneVideo", {
				aid,
				currentViews: stat.views,
				snapshotedAt: stat.time,
			}, { delay: 0 });
		}
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_AVAILABLE_PROXY") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:takeSnapshotForMilestoneVideoWorker",
			);
			SnapshotQueue.add("snapshotVideo", {
				aid: job.data.aid,
			}, { delay: 10 * SECOND });
			return;
		}
		throw e;
	} finally {
		client.release();
	}
};
