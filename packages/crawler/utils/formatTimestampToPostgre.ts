export function formatTimestampToPsql(timestamp: number) {
	const date = new Date(timestamp);
	return date.toISOString().slice(0, 23).replace("T", " ") + "+08";
}

export function parseTimestampFromPsql(timestamp: string) {
	return new Date(timestamp).getTime();
}
