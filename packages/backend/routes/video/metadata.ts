import { Elysia, t } from "elysia";
import { db, videoSnapshot } from "@core/drizzle";
import { bv2av } from "@backend/lib/bilibiliID";
import { getVideoInfo } from "@core/net/getVideoInfo";
import { redis } from "@core/db/redis";
import { ErrorResponseSchema } from "@backend/src/schema";
import type { VideoInfoData } from "@core/net/bilibili.d.ts";

export async function retrieveVideoInfoFromCache(aid: number) {
	const cacheKey = `cvsa:videoInfo:av${aid}`;
	const cachedData = await redis.get(cacheKey);
	if (cachedData) {
		return JSON.parse(cachedData);
	}
	return null;
}

async function setCache(aid: number, data: string) {
	const cacheKey = `cvsa:videoInfo:av${aid}`;
	return await redis.setex(cacheKey, 60, data);
}

async function insertVideoSnapshot(data: VideoInfoData) {
	const views = data.stat.view;
	const danmakus = data.stat.danmaku;
	const replies = data.stat.reply;
	const likes = data.stat.like;
	const coins = data.stat.coin;
	const shares = data.stat.share;
	const favorites = data.stat.favorite;
	const aid = data.aid;

	await db.insert(videoSnapshot).values({
		aid,
		views,
		danmakus,
		replies,
		likes,
		coins,
		shares,
		favorites
	});
}

export const getVideoMetadataHandler = new Elysia({ prefix: "/video" }).get(
	"/:id/info",
	async (c) => {
		const id = c.params.id;
		let aid: number | null = null;

		if (id.startsWith("BV1")) {
			aid = bv2av(id as `BV1${string}`);
		} else if (id.startsWith("av")) {
			aid = Number.parseInt(id.slice(2));
		} else {
			return c.status(400, {
				code: "MALFORMED_SLOT",
				message:
					"We cannot parse the video ID, or we currently do not support this format.",
				errors: []
			});
		}

		const cachedData = await retrieveVideoInfoFromCache(aid);
		if (cachedData) {
			return cachedData;
		}

		const r = await getVideoInfo(aid, "getVideoInfo");

		if (typeof r == "number") {
			return c.status(500, {
				code: "THIRD_PARTY_ERROR",
				message: `Got status code ${r} from bilibili API.`,
				errors: []
			});
		}

		const { data } = r;

		await setCache(aid, JSON.stringify(data));
		await insertVideoSnapshot(data);

		return data;
	},
	{
		response: {
			200: t.Any(),
			400: ErrorResponseSchema,
			500: ErrorResponseSchema
		}
	}
);
