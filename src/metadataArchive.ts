import arg from "arg";
import { Database } from "bun:sqlite";
import { getVideoDetails } from "@crawler/net/getVideoDetails";
import logger from "@core/log/logger";

const SECOND = 1000;
const SECONDS = SECOND;
const MINUTE = 60 * SECONDS;
const MINUTES = MINUTE;
const IPs = 6;

const rateLimits = [
	{ window: 5 * MINUTES, maxRequests: 160 * IPs },
	{ window: 30 * SECONDS, maxRequests: 20 * IPs },
	{ window: 1.2 * SECOND, maxRequests: 1 * IPs }
];

const requestQueue: number[] = [];

function isRateLimited(): boolean {
	const now = Date.now();
	return rateLimits.some(({ window, maxRequests }) => {
		const windowStart = now - window;
		const requestsInWindow = requestQueue.filter((timestamp) => timestamp >= windowStart).length;
		return requestsInWindow >= maxRequests;
	});
}

function logProgress(aid: number, processedAids: number, totalAids: number, startTime: number) {
	const elapsedTime = Date.now() - startTime;
	const elapsedSeconds = Math.floor(elapsedTime / 1000);
	const elapsedMinutes = Math.floor(elapsedSeconds / 60);
	const elapsedHours = Math.floor(elapsedMinutes / 60);

	const remainingAids = totalAids - processedAids;
	const averageTimePerAid = elapsedTime / processedAids;
	const eta = remainingAids * averageTimePerAid;
	const etaSeconds = Math.floor(eta / 1000);
	const etaMinutes = Math.floor(etaSeconds / 60);
	const etaHours = Math.floor(etaMinutes / 60);

	const progress = `${processedAids}/${totalAids}, ${((processedAids / totalAids) * 100).toFixed(
		2
	)}%, elapsed ${elapsedHours.toString().padStart(2, "0")}:${(elapsedMinutes % 60).toString().padStart(2, "0")}:${(
		elapsedSeconds % 60
	)
		.toString()
		.padStart(2, "0")}, ETA ${etaHours}h${(etaMinutes % 60).toString().padStart(2, "0")}m`;
	console.log(`Updated aid ${aid}, ${progress}`);
}

const quit = (reason: string) => {
	logger.error(reason);
	process.exit();
};

const args = arg({
	"--aids": String,
	"--db": String
});

const aidsFileName = args["--aids"];
const dbPath = args["--db"];

if (!aidsFileName) {
	quit("Missing --aids <file_path>");
}

if (!dbPath) {
	quit("Missing --db <path>");
}

const aidsFile = Bun.file(aidsFileName!);
const fileExists = await aidsFile.exists();

if (!fileExists) {
	quit(`${aidsFile} does not exist.`);
}

const aidsText = await aidsFile.text();
const aids = aidsText
	.split("\n")
	.map((line) => parseInt(line))
	.filter((num) => !Number.isNaN(num));

logger.log(`Read ${aids.length} aids.`);

const db = new Database(dbPath);
const existingAids = db.query<{ aid: number }, []>(`SELECT aid from bili_info_crawl`).all();
logger.log(`Existing Aids: ${existingAids.length}`);
const existingAidsSet = new Set(existingAids.map((a) => a.aid));
const newAids = aids.filter((aid) => !existingAidsSet.has(aid));
logger.log(`New Aids: ${newAids.length}`);

const totalAids = newAids.length;
let processedAids = 0;
const startTime = Date.now();

const processAid = async (aid: number) => {
	try {
		const data = await getVideoDetails(aid);
		if (data === null) {
			updateAidStatus(aid, "failed");
		} else {
			updateAidStatus(aid, "success", data.View.bvid, JSON.stringify(data));
		}
	} catch (error) {
		console.error(`Error updating aid ${aid}: ${error}`);
		updateAidStatus(aid, "failed");
	} finally {
		processedAids++;
		logProgress(aid, processedAids, totalAids, startTime);
	}
};

const interval = setInterval(async () => {
	if (newAids.length === 0) {
		clearInterval(interval);
		console.log("All aids processed.");
		return;
	}
	if (!isRateLimited()) {
		const aid = newAids.shift();
		if (aid !== undefined) {
			requestQueue.push(Date.now());
			await processAid(aid);
		}
	}
}, 50);

function updateAidStatus(aid: number, status: string, bvid?: string, data?: string) {
	const query = db.query(`
        INSERT INTO bili_info_crawl
		(aid, bvid, status, bvid, data, timestamp)
		VALUES ($aid, $bvid, $status, $bvid, $data, $timestamp)
    `);
	query.run({
		$aid: aid,
		$bvid: bvid || null,
		$status: status || null,
		$data: data || null,
		$timestamp: Date.now() / 1000
	});
}
