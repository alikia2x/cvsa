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
	if (!timeRangeHours ||timeRangeHours >= 7 * 24) {
		return 24 * 60;
	}
	if (interval >= 6 * HOUR) {
		return 6 * 60;
	} else if (interval >= 1 * HOUR) {
		return 60;
	} else if (interval >= 15 * MINUTE) {
		return 15;
	} else if (interval >= 5 * MINUTE) {
		return 5;
	}
	return 1;
};

export const processSnapshots = (snapshots: Snapshots | null, timeRangeHours?: number, timeOffsetHours: number = 0) => {
	if (!snapshots || snapshots.length === 0) {
		return [];
	}

	const sortedSnapshots = [...snapshots].sort(
		(a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
	);

	const oldestDate = new Date(sortedSnapshots[0].createdAt);
	const newestDate = new Date(sortedSnapshots[sortedSnapshots.length - 1].createdAt);

	// Calculate the time range with offset
	const targetEndTime = timeRangeHours ? new Date(newestDate.getTime() - timeOffsetHours * HOUR) : newestDate;
	const targetStartTime = timeRangeHours ? new Date(targetEndTime.getTime() - timeRangeHours * HOUR) : null;

	const startTime = targetStartTime ? (oldestDate > targetStartTime ? oldestDate : targetStartTime) : oldestDate;
	const endTime = targetEndTime;

	const hourlyTimePoints: Date[] = [];
	const currentTime = endTime;
	const timeDiff = currentTime.getTime() - startTime.getTime();
	const length = sortedSnapshots.filter((s) => {
		const snapshotTime = new Date(s.createdAt).getTime();
		return snapshotTime >= startTime.getTime() && snapshotTime <= endTime.getTime();
	}).length;
	const avgInterval = timeDiff / length;
	const dataIntervalMins = getDataIntervalMins(avgInterval, timeRangeHours);

	for (let time = new Date(startTime); time <= currentTime; time.setMinutes(time.getMinutes() + dataIntervalMins)) {
		hourlyTimePoints.push(new Date(time));
	}

	const processedData = hourlyTimePoints
		.map((timePoint) => {
			const previousSnapshots = sortedSnapshots.filter((s) => {
				const snapshotTime = new Date(s.createdAt).getTime();
				return snapshotTime <= timePoint.getTime() && snapshotTime >= startTime.getTime();
			});
	
			const nextSnapshots = sortedSnapshots.filter((s) => {
				const snapshotTime = new Date(s.createdAt).getTime();
				return snapshotTime >= timePoint.getTime() && snapshotTime <= endTime.getTime();
			});

			const previousSnapshot = previousSnapshots[previousSnapshots.length - 1];
			const nextSnapshot = nextSnapshots[0];

			if (!previousSnapshot && !nextSnapshot) {
				return null;
			}

			if (previousSnapshot && new Date(previousSnapshot.createdAt).getTime() === timePoint.getTime()) {
				return {
					createdAt: timePoint.toISOString(),
					views: previousSnapshot.views,
					likes: previousSnapshot.likes || 0,
					favorites: previousSnapshot.favorites || 0,
					coins: previousSnapshot.coins || 0,
					danmakus: previousSnapshot.danmakus || 0,
				};
			}

			if (previousSnapshot && !nextSnapshot) {
				return {
					createdAt: timePoint.toISOString(),
					views: previousSnapshot.views,
					likes: previousSnapshot.likes || 0,
					favorites: previousSnapshot.favorites || 0,
					coins: previousSnapshot.coins || 0,
					danmakus: previousSnapshot.danmakus || 0,
				};
			}

			if (!previousSnapshot && nextSnapshot) {
				return {
					createdAt: timePoint.toISOString(),
					views: nextSnapshot.views,
					likes: nextSnapshot.likes || 0,
					favorites: nextSnapshot.favorites || 0,
					coins: nextSnapshot.coins || 0,
					danmakus: nextSnapshot.danmakus || 0,
				};
			}

			const prevTime = new Date(previousSnapshot.createdAt).getTime();
			const nextTime = new Date(nextSnapshot.createdAt).getTime();
			const currentTime = timePoint.getTime();

			const ratio = (currentTime - prevTime) / (nextTime - prevTime);

			return {
				createdAt: timePoint.toISOString(),
				views: Math.round(previousSnapshot.views + (nextSnapshot.views - previousSnapshot.views) * ratio),
				likes: Math.round(
					(previousSnapshot.likes || 0) + ((nextSnapshot.likes || 0) - (previousSnapshot.likes || 0)) * ratio,
				),
				favorites: Math.round(
					(previousSnapshot.favorites || 0) +
						((nextSnapshot.favorites || 0) - (previousSnapshot.favorites || 0)) * ratio,
				),
				coins: Math.round(
					(previousSnapshot.coins || 0) + ((nextSnapshot.coins || 0) - (previousSnapshot.coins || 0)) * ratio,
				),
				danmakus: Math.round(
					(previousSnapshot.danmakus || 0) +
						((nextSnapshot.danmakus || 0) - (previousSnapshot.danmakus || 0)) * ratio,
				),
			};
		})
		.filter((d) => d !== null);

	return processedData;
};

export const detectMilestoneAchievements = (snapshots: Snapshots | null, publishedAt?: string): MilestoneAchievement[] => {
	if (!snapshots || snapshots.length < 2) {
		return [];
	}

	const sortedSnapshots = [...snapshots].sort(
		(a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
	);

	const milestones = [100000, 1000000, 10000000];
	const milestoneNames = ["殿堂", "传说", "神话"];
	const achievements: MilestoneAchievement[] = [];

	// Find the earliest snapshot for each milestone
	const earliestAchievements = new Map<number, MilestoneAchievement>();

	for (let i = 1; i < sortedSnapshots.length; i++) {
		const prevSnapshot = sortedSnapshots[i - 1];
		const currentSnapshot = sortedSnapshots[i];

		const prevTime = new Date(prevSnapshot.createdAt).getTime();
		const currentTime = new Date(currentSnapshot.createdAt).getTime();
		const timeDiff = currentTime - prevTime;

		// Check if snapshots are within 10 minutes
		if (timeDiff <= 10 * 60 * 1000) {
			for (let j = 0; j < milestones.length; j++) {
				const milestone = milestones[j];
				const milestoneName = milestoneNames[j];

				// Check if milestone was crossed between these two snapshots
				if (prevSnapshot.views < milestone && currentSnapshot.views >= milestone) {
					// Find the exact time when milestone was reached (linear interpolation)
					const ratio = (milestone - prevSnapshot.views) / (currentSnapshot.views - prevSnapshot.views);
					const milestoneTime = new Date(prevTime + ratio * timeDiff);

					const achievement: MilestoneAchievement = {
						milestone,
						milestoneName,
						achievedAt: milestoneTime.toISOString(),
						views: milestone,
					};

					// Only keep the earliest achievement for each milestone
					if (
						!earliestAchievements.has(milestone) ||
						new Date(achievement.achievedAt) < new Date(earliestAchievements.get(milestone)!.achievedAt)
					) {
						earliestAchievements.set(milestone, achievement);
					}
				}

				// Check if a snapshot exactly equals a milestone
				if (prevSnapshot.views === milestone || currentSnapshot.views === milestone) {
					const exactSnapshot = prevSnapshot.views === milestone ? prevSnapshot : currentSnapshot;
					const achievement: MilestoneAchievement = {
						milestone,
						milestoneName,
						achievedAt: exactSnapshot.createdAt,
						views: milestone,
					};

					if (
						!earliestAchievements.has(milestone) ||
						new Date(achievement.achievedAt) < new Date(earliestAchievements.get(milestone)!.achievedAt)
					) {
						earliestAchievements.set(milestone, achievement);
					}
				}
			}
		}
	}

	// Convert map to array and sort by milestone value
	const achievementsWithTime = Array.from(earliestAchievements.values()).sort((a, b) => a.milestone - b.milestone);

	// Calculate time taken for each achievement
	if (publishedAt) {
		const publishTime = new Date(publishedAt).getTime();
		
		for (const achievement of achievementsWithTime) {
			const achievementTime = new Date(achievement.achievedAt).getTime();
			const timeDiffMs = achievementTime - publishTime;
			
			// Convert to days, hours, minutes
			const days = Math.floor(timeDiffMs / (1000 * 60 * 60 * 24));
			const hours = Math.floor((timeDiffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
			const minutes = Math.floor((timeDiffMs % (1000 * 60 * 60)) / (1000 * 60));
			
			achievement.timeTaken = `${days} 天 ${hours} 时 ${minutes} 分`;
		}
	}

	return achievementsWithTime;
};
