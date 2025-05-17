import logger from "@core/log/logger.ts";
import { redis } from "@core/db/redis.ts";
import { sql } from "@core/db/dbNew.ts";
import { number, ValidationError } from "yup";
import { createHandlers } from "@/src/utils.ts";
import { getVideoInfo, getVideoInfoByBV } from "@core/net/getVideoInfo.ts";
import { idSchema } from "./snapshots.ts";
import { NetSchedulerError } from "@core/net/delegate.ts";
import type { Context } from "hono";
import type { BlankEnv, BlankInput } from "hono/types";
import type { VideoInfoData } from "@core/net/bilibili.d.ts";
import { startTime, endTime } from "hono/timing";

const CACHE_EXPIRATION_SECONDS = 60;

type ContextType = Context<BlankEnv, "/video/:id/info", BlankInput>;

async function insertVideoSnapshot(data: VideoInfoData) {
	const views = data.stat.view;
	const danmakus = data.stat.danmaku;
	const replies = data.stat.reply;
	const likes = data.stat.like;
	const coins = data.stat.coin;
	const shares = data.stat.share;
	const favorites = data.stat.favorite;
	const aid = data.aid;

	await sql`
        INSERT INTO video_snapshot (aid, views, danmakus, replies, likes, coins, shares, favorites)
        VALUES (${aid}, ${views}, ${danmakus}, ${replies}, ${likes}, ${coins}, ${shares}, ${favorites})
    `;

	logger.log(`Inserted into snapshot for video ${aid} by videoInfo API.`, "api", "fn:insertVideoSnapshot");
}

export const videoInfoHandler = createHandlers(async (c: ContextType) => {
	startTime(c, "parse", "Parse the request");
	try {
		const id = await idSchema.validate(c.req.param("id"));
		let videoId: string | number = id as string;
		if (videoId.startsWith("av")) {
			videoId = parseInt(videoId.slice(2));
		} else if (await number().isValid(videoId)) {
			videoId = parseInt(videoId);
		}

		const cacheKey = `cvsa:videoInfo:${videoId}`;
		endTime(c, "parse");
		startTime(c, "cache", "Check for cached data");
		const cachedData = await redis.get(cacheKey);
		endTime(c, "cache");
		if (cachedData) {
			return c.json(JSON.parse(cachedData));
		}
		startTime(c, "net", "Fetch data");
		let result: VideoInfoData | number;
		if (typeof videoId === "number") {
			result = await getVideoInfo(videoId, "getVideoInfo");
		} else {
			result = await getVideoInfoByBV(videoId, "getVideoInfo");
		}
		endTime(c, "net");

		if (typeof result === "number") {
			return c.json({ message: "Error fetching video info", code: result }, 500);
		}

		startTime(c, "db", "Write data to database");

		await redis.setex(cacheKey, CACHE_EXPIRATION_SECONDS, JSON.stringify(result));

		await insertVideoSnapshot(result);

		endTime(c, "db");
		return c.json(result);
	} catch (e) {
		if (e instanceof ValidationError) {
			return c.json({ message: "Invalid query parameters", errors: e.errors }, 400);
		} else if (e instanceof NetSchedulerError) {
			return c.json({ message: "Error fetching video info", code: e.code }, 500);
		} else {
			return c.json({ message: "Unhandled error", error: e }, 500);
		}
	}
});
