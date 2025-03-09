export const formatSeconds = (seconds: number) => {
	if (seconds < 60) {
		return `${(seconds).toFixed(1)}s`;
	}
	if (seconds < 3600) {
		return `${Math.floor(seconds / 60)}m${seconds % 60}s`;
	}
	return `${Math.floor(seconds / 3600)}h ${((seconds % 3600) / 60).toFixed(2)}m`;
};
