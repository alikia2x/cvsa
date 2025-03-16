import { Job } from "bullmq";
import { HOUR, MINUTE, SECOND } from "$std/datetime/constants.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { db } from "lib/db/init.ts";
import {
getIntervalFromLastSnapshotToNow,
	getShortTermEtaPrediction,
	getSongsNearMilestone,
	getUnsnapshotedSongs,
	songEligibleForMilestoneSnapshot,
} from "lib/db/snapshot.ts";
import { SnapshotQueue } from "lib/mq/index.ts";
import { insertVideoStats } from "lib/mq/task/getVideoStats.ts";
import { parseTimestampFromPsql } from "lib/utils/formatTimestampToPostgre.ts";
import { redis } from "lib/db/redis.ts";
import { NetSchedulerError } from "lib/mq/scheduler.ts";
import logger from "lib/log/logger.ts";
import { formatSeconds } from "lib/utils/formatSeconds.ts";
import { truncate } from "lib/utils/truncate.ts";

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

async function processMilestoneSnapshots(client: Client, vidoesNearMilestone: SongNearMilestone[]) {
	let i = 0;
	for (const snapshot of vidoesNearMilestone) {
		if (await snapshotScheduled(snapshot.aid)) {
			continue;
		}
		const timeFromLastSnapshot = await getIntervalFromLastSnapshotToNow(client, snapshot.aid);
		const lastSnapshotLessThan8Hrs = timeFromLastSnapshot && timeFromLastSnapshot * SECOND < 8 * HOUR;
		const notEligible = await songEligibleForMilestoneSnapshot(client, snapshot.aid);
		if (notEligible && lastSnapshotLessThan8Hrs) {
			continue;
		}
		const factor = Math.floor(i / 8);
		const delayTime = factor * SECOND * 2;
		await SnapshotQueue.add("snapshotMilestoneVideo", {
			aid: snapshot.aid,
			currentViews: snapshot.views,
			snapshotedAt: snapshot.created_at,
		}, { delay: delayTime, priority: 1 });
		await setSnapshotScheduled(snapshot.aid, true, 20 * 60);
		i++;
	}
}

async function processUnsnapshotedVideos(unsnapshotedVideos: number[]) {
	let i = 0;
	for (const aid of unsnapshotedVideos) {
		if (await snapshotScheduled(aid)) {
			logger.silly(`Video ${aid} is already scheduled for snapshot`, "mq", "fn:processUnsnapshotedVideos");
			continue;
		}
		const factor = Math.floor(i / 5);
		const delayTime = factor * SECOND * 4;
		await SnapshotQueue.add("snapshotVideo", {
			aid,
		}, { delay: delayTime, priority: 3 });
		await setSnapshotScheduled(aid, true, 6 * 60 * 60);
		i++;
	}
}

export const snapshotTickWorker = async (_job: Job) => {
	const client = await db.connect();
	try {
		const vidoesNearMilestone = await getSongsNearMilestone(client);
		await processMilestoneSnapshots(client, vidoesNearMilestone);

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
		const aid: number = job.data.aid;
		const currentViews: number = job.data.currentViews;
		const lastSnapshoted: string = job.data.snapshotedAt;
		const stat = await insertVideoStats(client, aid, "snapshotMilestoneVideo");
		if (typeof stat === "number") {
			if (stat === -404 || stat === 62002 || stat == 62012) {
				await setSnapshotScheduled(aid, true, 6 * 60 * 60);
			} else {
				await setSnapshotScheduled(aid, false, 0);
			}
			return;
		}
		const nextMilestone = currentViews >= 100000 ? 1000000 : 100000;
		if (stat.views >= nextMilestone) {
			await setSnapshotScheduled(aid, false, 0);
			return;
		}
		let eta = await getShortTermEtaPrediction(client, aid);
		if (eta === null) {
			const DELTA = 0.001;
			const intervalSeconds = (Date.now() - parseTimestampFromPsql(lastSnapshoted)) / SECOND;
			const viewsIncrement = stat.views - currentViews;
			const incrementSpeed = viewsIncrement / (intervalSeconds + DELTA);
			const viewsToIncrease = nextMilestone - stat.views;
			eta = viewsToIncrease / (incrementSpeed + DELTA);
		}
		const scheduledNextSnapshotDelay = eta * SECOND / 3;
		const maxInterval = 60 * MINUTE;
		const minInterval = 1 * SECOND;
		const delay = truncate(scheduledNextSnapshotDelay, minInterval, maxInterval);
		await SnapshotQueue.add("snapshotMilestoneVideo", {
			aid,
			currentViews: stat.views,
			snapshotedAt: stat.time,
		}, { delay, priority: 1 });
		await job.updateData({
			...job.data,
			updatedViews: stat.views,
			updatedTime: new Date(stat.time).toISOString(),
			etaInMins: eta / 60,
		});
		logger.log(
			`Scheduled next milestone snapshot for ${aid} in ${
				formatSeconds(delay / 1000)
			}, current views: ${stat.views}`,
			"mq",
		);
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_AVAILABLE_PROXY") {
			logger.warn(
				`No available proxy for aid ${job.data.aid}.`,
				"mq",
				"fn:takeSnapshotForMilestoneVideoWorker",
			);
			await SnapshotQueue.add("snapshotMilestoneVideo", {
				aid: job.data.aid,
				currentViews: job.data.currentViews,
				snapshotedAt: job.data.snapshotedAt,
			}, { delay: 5 * SECOND, priority: 1 });
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
		if (typeof stat === "number") {
			if (stat === -404 || stat === 62002 || stat == 62012) {
				await setSnapshotScheduled(aid, true, 6 * 60 * 60);
			} else {
				await setSnapshotScheduled(aid, false, 0);
			}
			return;
		}
		logger.log(`Taken snapshot for ${aid}`, "mq");
		if (stat == null) {
			setSnapshotScheduled(aid, false, 0);
			return;
		}
		await job.updateData({
			...job.data,
			updatedViews: stat.views,
			updatedTime: new Date(stat.time).toISOString(),
		});
		const nearMilestone = (stat.views >= 90000 && stat.views < 100000) ||
			(stat.views >= 900000 && stat.views < 1000000);
		if (nearMilestone) {
			await SnapshotQueue.add("snapshotMilestoneVideo", {
				aid,
				currentViews: stat.views,
				snapshotedAt: stat.time,
			}, { delay: 0, priority: 1 });
		}
		await setSnapshotScheduled(aid, false, 0);
	} catch (e) {
		if (e instanceof NetSchedulerError && e.code === "NO_AVAILABLE_PROXY") {
			await setSnapshotScheduled(job.data.aid, false, 0);
			return;
		}
		throw e;
	} finally {
		client.release();
	}
};
