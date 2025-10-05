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

async function fixMissingCover() {
	let fixQuery = "";
	let i = 0;
	let j = 0;
	const candidates = await pg`
        SELECT aid
        FROM
          bilibili_metadata
        WHERE
          cover_url IS NULL
    `;
	const query = sqlite.query(`SELECT data FROM bili_info_crawl WHERE aid = $aid`);
	for (const video of candidates) {
		j++;
		logger.log(`Progress: ${j}/${candidates.length}`);
		const aid: number = video.aid;
		try {
			const sqliteData: any = query.get({ $aid: aid });
			const rawData: VideoDetailsData | null = JSON.parse(sqliteData.data);
			if (!rawData) {
				logger.warn(`Data not exists for aid: ${aid}`);
				continue;
			}
			const coverURL = rawData.View.pic;
			if (!coverURL) continue;
			const q = `UPDATE bilibili_metadata SET cover_url = '${coverURL}' WHERE aid = ${aid};\n`;
			logger.log(`Fixing cover for aid: ${aid}`);
			i++;
			fixQuery += q;
		} catch (e) {
			//logger.error(e as Error, undefined, aid.toString());
			logger.error(aid.toString());
		}
		if (j % 1000 === 0) {
			const bytes = await Bun.write("scripts/fix_2.sql", fixQuery);
			logger.warn(`Wrote ${bytes} bytes`, "backup");
		}
	}
	logger.log(`Fixed ${i} videos, query length ${fixQuery.length}.`);
	return fixQuery;
}

const q = await fixMissingCover();
await Bun.write("scripts/fix_2.sql", q);
quit();
