import { Database } from "bun:sqlite";
import { sql } from "@core/index";
import logger from "@core/log";
import { getVideoDetails } from "@core/net/getVideoDetails";
import arg from "arg";

const quit = (reason?: string) => {
	reason && logger.error(reason);
	process.exit();
};

const args = arg({
	"--aids": String,
	"--db": String,
});

const dbPath = args["--db"];
if (!dbPath) {
	quit("Missing --db <path>");
}

const sqlite = new Database(dbPath);

const getAids = async () => {
	const aidsFile = args["--aids"];
	if (aidsFile) {
		return (await Bun.file(aidsFile).text()).split("\n").map(Number);
	}
	const aids = await sql<{ aid: number }[]>`SELECT aid FROM bilibili_metadata`;
	return aids.map((row: any) => row.aid);
};

async function addCandidates() {
	const aids = await getAids();

	logger.log(`Retrieved ${aids.length} from production DB.`);

	const existingAids = sqlite
		.prepare("SELECT aid FROM bili_info_crawl")
		.all()
		.map((row: any) => row.aid);
	logger.log(`We have ${existingAids.length} from local DB.`);

	const existingAidsSet = new Set(existingAids);

	const newAids = aids.filter((aid) => !existingAidsSet.has(aid));

	let i = 0;
	for (const aid of newAids) {
		const stmt = sqlite.query(
			`INSERT INTO bili_info_crawl (aid, status) VALUES ($aid, 'pending');`
		);
		stmt.all({ $aid: aid });
		i++;
		logger.log(`Added ${i} to local DB.`);
	}
	logger.log(`Added ${newAids.length} to local DB.`);
}

async function insertAidsToDB() {
	await addCandidates();

	const aidsInDB = sqlite
		.prepare("SELECT aid FROM bili_info_crawl WHERE status = 'pending'")
		.all()
		.map((row: any) => row.aid) as number[];

	const totalAids = aidsInDB.length;
	let processedAids = 0;
	const startTime = Date.now();

	const processAid = async (aid: number) => {
		try {
			const res = await getVideoDetails(aid);
			if (res === null) {
				updateAidStatus(aid, "failed");
			} else {
				updateAidStatus(aid, "success", res.View.bvid, JSON.stringify(res));
			}
		} catch (error) {
			console.error(`Error updating aid ${aid}: ${error}`);
			updateAidStatus(aid, "failed");
		} finally {
			processedAids++;
			logProgress(aid, processedAids, totalAids, startTime);
		}
	};

	const groupSize = 20;
	const groups = [];
	for (let i = 0; i < totalAids; i += groupSize) {
		groups.push(aidsInDB.slice(i, i + groupSize));
	}

	logger.log(`Processing ${totalAids} aids in ${groups.length} groups.`);

	for (const group of groups) {
		await Promise.all(group.map((aid) => processAid(aid)));
	}
}

function updateAidStatus(aid: number, status: string, bvid?: string, data?: string) {
	const stmt = sqlite.prepare(`
        UPDATE bili_info_crawl
        SET status = ?,
        ${bvid ? "bvid = ?," : ""}
        ${data ? "data = ?," : ""}
        timestamp = ?
        WHERE aid = ?
    `);
	const params = [
		status,
		...(bvid ? [bvid] : []),
		...(data ? [data] : []),
		Date.now() / 1000,
		aid,
	];
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

	const progress = `${processedAids}/${totalAids}, ${((processedAids / totalAids) * 100).toFixed(
		2
	)}%, elapsed ${elapsedHours.toString().padStart(2, "0")}:${(elapsedMinutes % 60).toString().padStart(2, "0")}:${(
		elapsedSeconds % 60
	)
		.toString()
		.padStart(2, "0")}, ETA ${etaHours}h${(etaMinutes % 60).toString().padStart(2, "0")}m`;
	logger.log(`Updated aid ${aid}, ${progress}`);
}

await insertAidsToDB();
quit();
