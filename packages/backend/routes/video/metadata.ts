import { Elysia, t } from "elysia";
import { db, videoSnapshot } from "@core/drizzle";
import { biliIDToAID, bv2av } from "@backend/lib/bilibiliID";
import { getVideoInfo } from "@core/net/getVideoInfo";
import { redis } from "@core/db/redis";
import { ErrorResponseSchema } from "@backend/src/schema";
import type { VideoInfoData } from "@core/net/bilibili.d.ts";
import { BiliAPIVideoMetadataSchema } from "@backend/lib/schema";

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
		const aid = biliIDToAID(id);

		if (!aid) {
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
		},
		detail: {
			summary: "Get video metadata",
			description:
				"This endpoint retrieves comprehensive metadata for a bilibili video. It accepts video IDs in av or BV format \
			and returns detailed information including title, description, uploader, statistics (views, likes, coins, etc.), \
			and publication date. The data is cached for 60 seconds to reduce API calls. If the video is not in cache, \
			it fetches fresh data from bilibili API and stores a snapshot in the database."
		}
	}
);
