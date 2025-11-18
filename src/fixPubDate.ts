import arg from "arg";
import { Database } from "bun:sqlite";
import logger from "@core/log";
import type { VideoDetailsData } from "@core/net/bilibili.d.ts";
import { sql } from "@core/index";

const quit = (reason?: string) => {
	reason && logger.error(reason);
	process.exit();
};

const args = arg({
	"--db": String
});

const dbPath = args["--db"];
if (!dbPath) {
	quit("Missing --db <path>");
}

const sqlite = new Database(dbPath);
const pg = sql;

async function fixTimezoneError() {
	let fixQuery = "";
	let i = 0;
	let j = 0;
	const candidates = await pg`
        SELECT aid, published_at
        FROM
          songs
        WHERE
          published_at <= '2000-01-01'
    `;
	const query = sqlite.query(`SELECT data FROM bili_info_crawl WHERE aid = $aid`);
	for (const video of candidates) {
		const aid: number = video.aid;
		try {
			const sqliteData: any = query.get({ $aid: aid });
			const rawData: VideoDetailsData | null = JSON.parse(sqliteData.data);
			if (!rawData) {
				logger.warn(`Data not exists for aid: ${aid}`);
				continue;
			}
			const realTimestamp = rawData.View.pubdate;
			const dbTimestamp = video.published_at.getTime() / 1000;
			const diff = dbTimestamp - realTimestamp;
			if (Math.abs(diff) > 1) {
				logger.warn(`Find incorrect timestamp for aid ${aid} with diff of ${diff} sec`);
				const date = new Date(realTimestamp * 1000).toISOString();
				const q = `UPDATE bilibili_metadata SET published_at = '${date}' WHERE aid = ${aid};\n`;
				fixQuery += q;
				i++;
			}
		} catch (e) {
			//logger.error(e as Error, undefined, aid.toString());
			logger.error(aid.toString());
		}
		j++;
		logger.log(`Progress: ${j}/${candidates.length}`);
	}
	logger.log(`Fixed ${i} videos, query length ${fixQuery.length}.`);
	return fixQuery;
}

const q = await fixTimezoneError();
await Bun.write("scripts/fix.sql", q);
quit();
