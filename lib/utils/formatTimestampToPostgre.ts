export default function formatTimestamp(timestamp: number) {
	const date = new Date(timestamp * 1000);
	return date.toISOString().slice(0, 19).replace("T", " ");
}