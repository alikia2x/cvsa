import path from "node:path";
import { Database } from "jsr:@db/sqlite@0.12";
import { getBiliBiliVideoInfo } from "./videoInfo.ts";
import { ensureDir } from "https://deno.land/std@0.113.0/fs/mod.ts";

const aidPath = "./data/2025010104_c30_aids.txt";
const db = new Database("./data/main.db", { int64: true });
const regions = ["shanghai", "hangzhou", "qingdao", "beijing", "zhangjiakou", "chengdu", "shenzhen", "hohhot"];
const logDir = "./logs/bili-info-crawl";
const logFile = path.join(logDir, `run-${Date.now() / 1000}.log`);
const shouldReadTextFile = false;

const SECOND = 1000;
const SECONDS = SECOND;
const MINUTE = 60 * SECONDS;
const MINUTES = MINUTE;
const IPs = regions.length;

const rateLimits = [
	{ window: 5 * MINUTES, maxRequests: 160 * IPs },
	{ window: 30 * SECONDS, maxRequests: 20 * IPs },
	{ window: 1.2 * SECOND, maxRequests: 1 * IPs },
];

const requestQueue: number[] = [];

async function setupLogging() {
	await ensureDir(logDir);
	const logStream = await Deno.open(logFile, { write: true, create: true, append: true });

	const redirectConsole =
		// deno-lint-ignore no-explicit-any
		(originalConsole: (...args: any[]) => void) =>
		// deno-lint-ignore no-explicit-any
		(...args: any[]) => {
			const message = args.map((arg) => (typeof arg === "object" ? JSON.stringify(arg) : arg)).join(" ");
			originalConsole(message);
			logStream.write(new TextEncoder().encode(message + "\n"));
		};

	console.log = redirectConsole(console.log);
	console.error = redirectConsole(console.error);
	console.warn = redirectConsole(console.warn);
}

function isRateLimited(): boolean {
	const now = Date.now();
	return rateLimits.some(({ window, maxRequests }) => {
		const windowStart = now - window;
		const requestsInWindow = requestQueue.filter((timestamp) => timestamp >= windowStart).length;
		return requestsInWindow >= maxRequests;
	});
}

async function readFromText() {
	const aidRawcontent = await Deno.readTextFile(aidPath);
	const aids = aidRawcontent
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => parseInt(line));

	// if (!db.prepare("SELECT COUNT(*) FROM bili_info_crawl").get()) {
	//     const insertStmt = db.prepare("INSERT OR IGNORE INTO bili_info_crawl (aid, status) VALUES (?, 'pending')");
	//     aids.forEach((aid) => insertStmt.run(aid));
	// }

	// 查询数据库中已经存在的 aid
	const existingAids = db
		.prepare("SELECT aid FROM bili_info_crawl")
		.all()
		.map((row) => row.aid);
	console.log(existingAids.length);

	// 将 existingAids 转换为 Set 以提高查找效率
	const existingAidsSet = new Set(existingAids);

	// 找出 aids 数组中不存在于数据库的条目
	const newAids = aids.filter((aid) => !existingAidsSet.has(aid));

	// 插入这些新条目
	const insertStmt = db.prepare("INSERT OR IGNORE INTO bili_info_crawl (aid, status) VALUES (?, 'pending')");
	newAids.forEach((aid) => insertStmt.run(aid));
}

async function insertAidsToDB() {
	if (shouldReadTextFile) {
		await readFromText();
	}

	const aidsInDB = db
		.prepare("SELECT aid FROM bili_info_crawl WHERE status = 'pending' OR status = 'failed'")
		.all()
		.map((row) => row.aid) as number[];

	const totalAids = aidsInDB.length;
	let processedAids = 0;
	const startTime = Date.now();

	const processAid = async (aid: number) => {
		try {
			const res = await getBiliBiliVideoInfo(aid, regions[processedAids % regions.length]);
			if (res === null) {
				updateAidStatus(aid, "failed");
			} else {
				const rawData = JSON.parse(res);
				if (rawData.code === 0) {
					updateAidStatus(aid, "success", rawData.data.View.bvid, JSON.stringify(rawData.data));
				} else {
					updateAidStatus(aid, "error", undefined, res);
				}
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
		if (aidsInDB.length === 0) {
			clearInterval(interval);
			console.log("All aids processed.");
			return;
		}
		if (!isRateLimited()) {
			const aid = aidsInDB.shift();
			if (aid !== undefined) {
				requestQueue.push(Date.now());
				await processAid(aid);
			}
		}
	}, 50);

	console.log("Starting to process aids...");
}

function updateAidStatus(aid: number, status: string, bvid?: string, data?: string) {
	const stmt = db.prepare(`
        UPDATE bili_info_crawl
        SET status = ?,
        ${bvid ? "bvid = ?," : ""}
        ${data ? "data = ?," : ""}
        timestamp = ?
        WHERE aid = ?
    `);
	const params = [status, ...(bvid ? [bvid] : []), ...(data ? [data] : []), Date.now() / 1000, aid];
	stmt.run(...params);
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

	const progress = `${processedAids}/${totalAids}, ${
		((processedAids / totalAids) * 100).toFixed(
			2,
		)
	}%, elapsed ${elapsedHours.toString().padStart(2, "0")}:${(elapsedMinutes % 60).toString().padStart(2, "0")}:${
		(
			elapsedSeconds % 60
		)
			.toString()
			.padStart(2, "0")
	}, ETA ${etaHours}h${(etaMinutes % 60).toString().padStart(2, "0")}m`;
	console.log(`Updated aid ${aid}, ${progress}`);
}

await setupLogging();
insertAidsToDB();
