import { log } from "@core/lib";

export const getMileStoneETAfactor = (x: number) => {
	const a = 1.054;
	const b = 4.5;
	const c = 100;
	const u = 0.601;
	const g = 455;
	if (x > g) {
		return log(b / log(x + 1), a);
	} else {
		return log(b / log(x + c), a) + u;
	}
};

export const getClosetMilestone = (views: number) => {
	if (views < 100000) return 100000;
	if (views < 1000000) return 1000000;
	return Math.ceil(views / 1000000) * 1000000;
};
