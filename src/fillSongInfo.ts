import arg from "arg";
import logger from "@core/log";
import { sql } from "@core/index";
import type { Row } from "postgres";

const quit = (reason?: string) => {
	reason && logger.error(reason);
	process.exit();
};

const args = arg({
	"--file": String
});

const dataPath = args["--file"];
if (!dataPath) {
	quit("Missing --file <path>");
}

interface Item {
	name: string;
	singer: string[];
}

type DataFile = {
	[key: string]: Item;
};

const pg = sql;

async function getVideoInfo(id: string): Promise<Row | undefined> {
	if (parseInt(id)) {
		return (
			await pg`
			SELECT aid, bvid
			FROM bilibili_metadata
			WHERE aid = ${id}
		`
		)[0];
	} else if (id.startsWith("av")) {
		return (
			await pg`
			SELECT aid, bvid
			FROM bilibili_metadata
			WHERE aid = ${id.replace("av", "")}
		`
		)[0];
	} else if (id.startsWith("BV")) {
		return (
			await pg`
			SELECT aid, bvid
			FROM bilibili_metadata
			WHERE bvid = ${id}
		`
		)[0];
	} else {
		return undefined;
	}
}

async function getSingerID(name: string): Promise<number | undefined> {
	const singer = await pg`
		SELECT id
		FROM singer
		WHERE name = ${name}
	`;
	if (singer.length > 0) {
		return singer[0]?.id;
	}
	const singerID = await pg`
		INSERT INTO singer (name)
		VALUES (${name})
		RETURNING id
	`;
	return singerID[0]?.id;
}

async function processVideo(key: string, item: Item) {
	const videoInfo = await getVideoInfo(key);
	if (!videoInfo) {
		logger.warn(`Video not found: ${key}`);
		return;
	}
	const aid = videoInfo.aid;
	await pg`
		UPDATE songs
		SET name = ${item.name}
		WHERE aid = ${aid}
	`;
	const singerIDs = (await Promise.all(item.singer.map(async (singer) => await getSingerID(singer)))).filter(
		(id) => id !== undefined
	);
	for (const singerID of singerIDs) {
		await pg`
			INSERT INTO
			relations (source_id, source_type, target_id, target_type, relation)
			VALUES (${aid}, 'song', ${singerID}, 'singer', 'sing')
			ON CONFLICT (source_id, source_type, target_id, target_type, relation) DO NOTHING;
		`;
	}
}

async function fillSongInfo() {
	let fixQuery = "";
	let i = 0;
	const file = Bun.file(dataPath!);
	const candidates: DataFile = await file.json();
	const length = Object.keys(candidates).length;
	for (const videoID in candidates) {
		await processVideo(videoID, candidates[videoID]!);
		i++;
		logger.log(`Progress: ${i}/${length}`);
	}
	return fixQuery;
}

await fillSongInfo();
quit();
