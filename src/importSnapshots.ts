import logger from "@core/log";
import { sql } from "@core/index";
import arg from "arg";
import fs from "fs/promises";
import path from "path";

const quit = (reason?: string) => {
	reason && logger.error(reason);
	process.exit();
};

interface Record {
	id: number;
	added: number;
	aid: number;
	view: number;
	danmaku: number;
	reply: number;
	favorite: number;
	coin: number;
	share: number;
	like: number;
	dislike: number | null;
	now_rank: number | null;
	his_rank: number | null;
	vt: number | null;
	vv: number | null;
}

async function fetchData(aid: number): Promise<Record[]> {
	const cacheDir = path.resolve("temp/tdd");
	const cacheFile = path.join(cacheDir, `${aid}.json`);
	console.log(cacheFile)
	try {
		const cached = await fs.readFile(cacheFile, "utf-8");
		logger.log(`Using cached data for aid ${aid}`);
		return JSON.parse(cached) as Record[];
	} catch (e){
		console.error(e)
		logger.log(`Fetching data from API for aid ${aid}`);
		const url = `https://api.bunnyxt.com/tdd/v2/video/${aid}/record`;
		const res = await fetch(url);
		if (!res.ok) {
			throw new Error(`Failed to fetch data: ${res.status} ${res.statusText}`);
		}
		const data = (await res.json()) as Record[];

		await fs.mkdir(cacheDir, { recursive: true });
		await fs.writeFile(cacheFile, JSON.stringify(data, null, 2), "utf-8");

		return data;
	}
}

const args = arg({
	"--aid": Number
});

const aid = args["--aid"];
if (!aid) {
	quit("Missing --aid <aid>");
}

const pg = sql;

async function importData() {
	const data = await fetchData(aid!);
	const length = data.length;
	logger.log(`Found ${length} snapshots for aid ${aid}`);
	let i = 0;
	for (const record of data) {
		try {
			const time = new Date(record.added * 1000);
			const timeString = time.toISOString().replace("T", " ");
			await pg`
                INSERT INTO video_snapshot (aid, created_at, views, danmakus, replies, favorites, coins, shares, likes)
                VALUES (${record.aid}, ${timeString}, ${record.view}, ${record.danmaku}, ${record.reply}, ${record.favorite}, ${record.coin}, ${record.share}, ${record.like})
            `;
		} catch (e) {
			logger.error(e as Error);
			logger.warn(
				`Failed to import snapshot for aid ${record.aid} at ${record.added}, id: ${record.id}`
			);
		}
		i++;
		logger.log(`Importing snapshots for aid ${record.aid} - Progress: ${i}/${length}`);
	}
}

await importData();
quit();
