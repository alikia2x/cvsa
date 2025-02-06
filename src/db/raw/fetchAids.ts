import { Database } from "jsr:@db/sqlite@0.12";
import { ensureDir } from "https://deno.land/std@0.113.0/fs/mod.ts";

// 常量定义
const MAX_RETRIES = 3;
const API_URL = "https://api.bilibili.com/x/web-interface/newlist?rid=30&ps=50&pn=";
const DATABASE_PATH = "./data/main.db";
const LOG_DIR = "./logs/bili-info-crawl";
const LOG_FILE = `${LOG_DIR}/run-${Date.now() / 1000}.log`;

// 打开数据库
const db = new Database(DATABASE_PATH, { int64: true });

// 设置日志
async function setupLogging() {
	await ensureDir(LOG_DIR);
	const logStream = await Deno.open(LOG_FILE, { write: true, create: true, append: true });

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

interface Metadata {
	key: string;
	value: string;
}

// 获取最后一次更新的时间
function getLastUpdate(): Date {
	const result = db.prepare("SELECT value FROM metadata WHERE key = 'fetchAid-lastUpdate'").get() as Metadata;
	return result ? new Date(result.value as string) : new Date(0);
}

// 更新最后更新时间
function updateLastUpdate() {
	const now = new Date().toISOString();
	db.prepare("UPDATE metadata SET value = ? WHERE key = 'fetchAid-lastUpdate'").run(now);
}

// 辅助函数：获取数据
// deno-lint-ignore no-explicit-any
async function fetchData(pn: number, retries = MAX_RETRIES): Promise<any> {
	try {
		const response = await fetch(`${API_URL}${pn}`);
		if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
		return await response.json();
	} catch (error) {
		if (retries > 0) {
			await new Promise((resolve) => setTimeout(resolve, 1000));
			return fetchData(pn, retries - 1);
		}
		throw error;
	}
}

// 插入 aid 到数据库
function insertAid(aid: number) {
	db.prepare("INSERT OR IGNORE INTO bili_info_crawl (aid, status) VALUES (?, 'pending')").run(aid);
}

// 主函数
async function main() {
	await setupLogging();

	let pn = 1;
	let shouldContinue = true;
	const lastUpdate = getLastUpdate();

	while (shouldContinue) {
		try {
			const data = await fetchData(pn);
			const archives = data.data.archives;

			for (const archive of archives) {
				const pubTime = new Date(archive.pubdate * 1000);
				if (pubTime > lastUpdate) {
					insertAid(archive.aid);
				} else {
					shouldContinue = false;
					break;
				}
			}

			pn++;
			console.log(`Fetched page ${pn}`);
		} catch (error) {
			console.error(`Error fetching data for pn=${pn}: ${error}`);
		}
	}

	// 更新最后更新时间
	updateLastUpdate();

	// 关闭数据库
	db.close();
}

// 运行主函数
main().catch(console.error);
