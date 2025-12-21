import { HOUR, MINUTE } from "@core/lib";
import type { Snapshots } from "./index";

export interface MilestoneAchievement {
	milestone: number;
	milestoneName: string;
	achievedAt: string;
	views: number;
	timeTaken?: string;
}

const getDataIntervalMins = (interval: number, timeRangeHours?: number) => {
	if (!timeRangeHours || timeRangeHours > 90 * 24) {
		return 24 * 60;
	} else if (interval >= 6 * HOUR) {
		return 6 * 60;
	} else if (interval >= HOUR) {
		return 60;
	} else if (interval >= 15 * MINUTE) {
		return 15;
	} else if (interval >= 5 * MINUTE) {
		return 5;
	}
	return 1;
};

export const processSnapshots = (
	snapshotTimestamps: (Exclude<Snapshots, null>[number] & { timestamp: number })[] | null,
	timeRangeHours?: number,
	timeOffsetHours: number = 0
) => {
	if (!snapshotTimestamps || snapshotTimestamps.length === 0) {
		return [];
	}

	const oldestTimestamp = snapshotTimestamps[0].timestamp;
	const newestTimestamp = snapshotTimestamps[snapshotTimestamps.length - 1].timestamp;

	const targetEndTime = timeRangeHours
		? newestTimestamp - timeOffsetHours * HOUR
		: newestTimestamp;
	const targetStartTime = timeRangeHours
		? targetEndTime - timeRangeHours * HOUR
		: oldestTimestamp;

	const startTime = Math.max(oldestTimestamp, targetStartTime);
	const endTime = targetEndTime;

	const beforeRangeSnapshots = snapshotTimestamps.filter((s) => s.timestamp < startTime);
	const afterRangeSnapshots = snapshotTimestamps.filter((s) => s.timestamp > endTime);

	const closestBefore =
		beforeRangeSnapshots.length > 0
			? beforeRangeSnapshots[beforeRangeSnapshots.length - 1]
			: null;
	const closestAfter = afterRangeSnapshots.length > 0 ? afterRangeSnapshots[0] : null;

	const relevantSnapshots = snapshotTimestamps.filter(
		(s) => s.timestamp >= startTime && s.timestamp <= endTime
	);

	if (relevantSnapshots.length === 0) {
		if (!closestBefore && !closestAfter) {
			return [];
		}

		if (closestBefore && !closestAfter) {
			return [createSnapshotData(startTime, closestBefore)];
		}
		if (!closestBefore && closestAfter) {
			return [createSnapshotData(startTime, closestAfter)];
		}

		if (closestBefore && closestAfter) {
			const timeDiff = closestAfter.timestamp - closestBefore.timestamp;
			const ratio = (startTime - closestBefore.timestamp) / timeDiff;
			return [createInterpolatedSnapshot(startTime, closestBefore, closestAfter, ratio)];
		}
	}

	const timeDiff = endTime - startTime;
	const avgInterval = timeDiff / Math.max(relevantSnapshots.length, 1);
	const dataIntervalMins = getDataIntervalMins(avgInterval, timeRangeHours);
	const dataIntervalMs = dataIntervalMins * 60 * 1000;

	const hourlyTimePoints: number[] = [];
	for (let time = startTime; time <= endTime; time += dataIntervalMs) {
		hourlyTimePoints.push(time);
	}

	let snapshotIndex = 0;
	const processedData = [];

	for (const timePoint of hourlyTimePoints) {
		while (
			snapshotIndex < relevantSnapshots.length - 1 &&
			relevantSnapshots[snapshotIndex].timestamp < timePoint
		) {
			snapshotIndex++;
		}

		const currentSnapshot = relevantSnapshots[snapshotIndex];
		const prevSnapshot = snapshotIndex > 0 ? relevantSnapshots[snapshotIndex - 1] : null;

		let result = null;

		if (currentSnapshot && currentSnapshot.timestamp === timePoint) {
			result = createSnapshotData(timePoint, currentSnapshot);
		} else if (prevSnapshot && currentSnapshot && prevSnapshot.timestamp <= timePoint) {
			const ratio =
				(timePoint - prevSnapshot.timestamp) /
				(currentSnapshot.timestamp - prevSnapshot.timestamp);
			result = createInterpolatedSnapshot(timePoint, prevSnapshot, currentSnapshot, ratio);
		} else if (!prevSnapshot && currentSnapshot && currentSnapshot.timestamp >= timePoint) {
			result = createSnapshotData(timePoint, currentSnapshot);
		} else if (
			snapshotIndex === relevantSnapshots.length - 1 &&
			currentSnapshot &&
			currentSnapshot.timestamp <= timePoint
		) {
			result = createSnapshotData(timePoint, currentSnapshot);
		} else {
			if (closestBefore && closestAfter) {
				const timeDiff = closestAfter.timestamp - closestBefore.timestamp;
				const ratio = (timePoint - closestBefore.timestamp) / timeDiff;
				result = createInterpolatedSnapshot(timePoint, closestBefore, closestAfter, ratio);
			} else if (closestBefore) {
				result = createSnapshotData(timePoint, closestBefore);
			} else if (closestAfter) {
				result = createSnapshotData(timePoint, closestAfter);
			}
		}

		if (result) {
			processedData.push(result);
		}
	}

	return processedData;
};

const createSnapshotData = (timestamp: number, snapshot: any) => ({
	createdAt: new Date(timestamp).toISOString(),
	views: snapshot.views,
	likes: snapshot.likes || 0,
	favorites: snapshot.favorites || 0,
	coins: snapshot.coins || 0,
	danmakus: snapshot.danmakus || 0,
});

const createInterpolatedSnapshot = (timestamp: number, prev: any, next: any, ratio: number) => ({
	createdAt: new Date(timestamp).toISOString(),
	views: Math.round(prev.views + (next.views - prev.views) * ratio),
	likes: Math.round((prev.likes || 0) + ((next.likes || 0) - (prev.likes || 0)) * ratio),
	favorites: Math.round(
		(prev.favorites || 0) + ((next.favorites || 0) - (prev.favorites || 0)) * ratio
	),
	coins: Math.round((prev.coins || 0) + ((next.coins || 0) - (prev.coins || 0)) * ratio),
	danmakus: Math.round(
		(prev.danmakus || 0) + ((next.danmakus || 0) - (prev.danmakus || 0)) * ratio
	),
});

export const detectMilestoneAchievements = (
	snapshots: Snapshots | null,
	publishedAt?: string
): MilestoneAchievement[] => {
	if (!snapshots || snapshots.length < 2) {
		return [];
	}

	const sortedSnapshots = [...snapshots].sort(
		(a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
	);

	const milestones = [100000, 1000000, 10000000];
	const milestoneNames = ["殿堂", "传说", "神话"];

	const earliestAchievements = new Map<number, MilestoneAchievement>();

	for (let i = 1; i < sortedSnapshots.length; i++) {
		const prevSnapshot = sortedSnapshots[i - 1];
		const currentSnapshot = sortedSnapshots[i];

		const prevTime = new Date(prevSnapshot.createdAt).getTime();
		const currentTime = new Date(currentSnapshot.createdAt).getTime();
		const timeDiff = currentTime - prevTime;

		if (timeDiff <= 10 * 60 * 1000) {
			for (let j = 0; j < milestones.length; j++) {
				const milestone = milestones[j];
				const milestoneName = milestoneNames[j];

				if (prevSnapshot.views < milestone && currentSnapshot.views >= milestone) {
					const ratio =
						(milestone - prevSnapshot.views) /
						(currentSnapshot.views - prevSnapshot.views);
					const milestoneTime = new Date(prevTime + ratio * timeDiff);

					const achievement: MilestoneAchievement = {
						milestone,
						milestoneName,
						achievedAt: milestoneTime.toISOString(),
						views: milestone,
					};

					if (
						!earliestAchievements.has(milestone) ||
						new Date(achievement.achievedAt) <
							new Date(earliestAchievements.get(milestone)!.achievedAt)
					) {
						earliestAchievements.set(milestone, achievement);
					}
				}

				if (prevSnapshot.views === milestone || currentSnapshot.views === milestone) {
					const exactSnapshot =
						prevSnapshot.views === milestone ? prevSnapshot : currentSnapshot;
					const achievement: MilestoneAchievement = {
						milestone,
						milestoneName,
						achievedAt: exactSnapshot.createdAt,
						views: milestone,
					};

					if (
						!earliestAchievements.has(milestone) ||
						new Date(achievement.achievedAt) <
							new Date(earliestAchievements.get(milestone)!.achievedAt)
					) {
						earliestAchievements.set(milestone, achievement);
					}
				}
			}
		}
	}

	const achievementsWithTime = Array.from(earliestAchievements.values()).sort(
		(a, b) => a.milestone - b.milestone
	);

	if (publishedAt) {
		const publishTime = new Date(publishedAt).getTime();

		for (const achievement of achievementsWithTime) {
			const achievementTime = new Date(achievement.achievedAt).getTime();
			const timeDiffMs = achievementTime - publishTime;

			const days = Math.floor(timeDiffMs / (1000 * 60 * 60 * 24));
			const hours = Math.floor((timeDiffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
			const minutes = Math.floor((timeDiffMs % (1000 * 60 * 60)) / (1000 * 60));

			achievement.timeTaken = `${days} 天 ${hours} 时 ${minutes} 分`;
		}
	}

	return achievementsWithTime;
};
