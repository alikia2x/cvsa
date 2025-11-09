import { HOUR, MINUTE } from "@core/lib";
import type { Snapshots } from "./index";

const getDataIntervalMins = (interval: number) => {
	if (interval >= 6 * HOUR) {
		return 6 * 60;
	}
	else if (interval >= 1 * HOUR) {
		return 60;
	}
	else if (interval >= 15 * MINUTE) {
		return 15;
	}
	else if (interval >= 5 * MINUTE) {
		return 5;
	}
	return 1;
}

export const processSnapshots = (snapshots: Snapshots | null, timeRangeHours: number = 14 * 24) => {
	if (!snapshots || snapshots.length === 0) {
		return [];
	}

	const sortedSnapshots = [...snapshots].sort(
		(a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
	);

	const oldestDate = new Date(sortedSnapshots[0].createdAt);

	const targetStartTime = new Date(new Date().getTime() - timeRangeHours * HOUR);

	const startTime = oldestDate > targetStartTime ? oldestDate : targetStartTime;

	const hourlyTimePoints: Date[] = [];
	const currentTime = new Date(sortedSnapshots[sortedSnapshots.length - 1].createdAt);
	const timeDiff = currentTime.getTime() - startTime.getTime();
	const length = sortedSnapshots.filter((s) => new Date(s.createdAt) >= startTime).length;
	const avgInterval = timeDiff / length;
	const dataIntervalMins = getDataIntervalMins(avgInterval);

	for (let time = new Date(startTime); time <= currentTime; time.setMinutes(time.getMinutes() + dataIntervalMins)) {
		hourlyTimePoints.push(new Date(time));
	}

	const processedData = hourlyTimePoints
		.map((timePoint) => {
			const previousSnapshots = sortedSnapshots.filter((s) => new Date(s.createdAt) <= timePoint);

			const nextSnapshots = sortedSnapshots.filter((s) => new Date(s.createdAt) >= timePoint);

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