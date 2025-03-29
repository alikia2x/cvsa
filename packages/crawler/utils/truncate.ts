export function truncate(num: number, min: number, max: number) {
	return Math.max(min, Math.min(num, max));
}
