import { bilibiliMetadata, db, eta, type VideoSnapshotType } from "@core/drizzle";
import { SECOND } from "@core/lib";
import logger from "@core/log";
import { NetSchedulerError } from "@core/net/delegate";
import { bulkGetVideoStats } from "@crawler/net/bulkGetVideoStats";
import { desc, eq } from "drizzle-orm";

const store = Bun.file(`temp/annualSnapshots.json`);

const snapshots: Omit<VideoSnapshotType, "id">[] = [];

if (await store.exists()) {
	// load
}

const aids = await db
	.select({
		aid: bilibiliMetadata.aid,
	})
	.from(bilibiliMetadata)
	.leftJoin(eta, eq(bilibiliMetadata.aid, eta.aid))
	.orderBy(desc(eta.speed))
	.then((rows) => {
		const mapped = rows.map((row) => row.aid);
		return mapped.filter((item): item is number => item !== null);
	});

const totalAids = aids.length;

logger.log(`Total aids: ${totalAids}`);

const bulkSize = 50;

const groupedAids = [];
for (let i = 0; i < aids.length; i += bulkSize) {
	groupedAids.push(aids.slice(i, i + bulkSize));
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const serialize = async () => {
	const json = JSON.stringify(snapshots, null, 4);
	await store.write(json);
};

let aidsProcessed = 0;

const requestForSnapshot = async (aids: number[], depth: number = 0) => {
	if (depth > 10) {
		logger.error(`Cannot fetch metadata for aids: ${aids.join(",")}, depth: ${depth}`);
		return;
	}
	try {
		const rawData = await bulkGetVideoStats(aids, "annualArchive");
		logger.log(`Fetched metadata for ${aids.length} aids, depth: ${depth}`);
		if (typeof rawData === "number") {
			await sleep(2 * SECOND);
			return requestForSnapshot(aids, depth + 1);
		}
		for (const item of rawData.data) {
			snapshots.push({
				aid: item.id,
				coins: item.cnt_info.coin,
				createdAt: new Date(rawData.time).toISOString(),
				danmakus: item.cnt_info.danmaku,
				favorites: item.cnt_info.collect,
				likes: item.cnt_info.thumb_up,
				replies: item.cnt_info.reply,
				shares: item.cnt_info.share,
				views: item.cnt_info.play,
			});
			aidsProcessed += 1;
		}
	} catch (e) {
		if (e instanceof NetSchedulerError) {
			requestForSnapshot(aids, 1);
		}
	}
};

const taskFactories = groupedAids.map((group) => () => requestForSnapshot(group));

const concurrency = 100;

for (let i = 0; i < taskFactories.length; i += concurrency) {
	const batch = taskFactories.slice(i, i + concurrency);

	await Promise.all(batch.map((factory) => factory()));

	logger.log(`Processed ${aidsProcessed} of ${totalAids}`);
	await sleep(1.7 * SECOND);
	serialize();
}
await serialize();
process.exit(0);
