import { biliIDToAID } from "@backend/lib/bilibiliID";
import { BiliAPIVideoMetadataSchema, BiliVideoSchema, SongSchema } from "@backend/lib/schema";
import { redis } from "@core/db/redis";
import { bilibiliMetadata, db, latestVideoSnapshot, songs } from "@core/drizzle";
import type { VideoInfoData } from "@core/net/bilibili";
import { getVideoInfo } from "@core/net/getVideoInfo";
import { eq, ilike } from "drizzle-orm";
import { Elysia } from "elysia";
import { z } from "zod";
import { retrieveVideoInfoFromCache } from "../video/metadata";

const getSongSearchResult = async (searchQuery: string) => {
	const data = await db
		.select()
		.from(songs)
		.innerJoin(latestVideoSnapshot, eq(songs.aid, latestVideoSnapshot.aid))
		.where(ilike(songs.name, `%${searchQuery}%`));

	const results = data
		.map((song) => {
			if (!song.songs.name) return null;
			const occurrences = (song.songs.name.match(new RegExp(searchQuery, "gi")) || []).length;
			const lengthRatio = searchQuery.length / song.songs.name.length;
			const viewsLog = Math.log10(song.latest_video_snapshot.views + 1);
			return {
				data: song,
				lengthRatio,
				occurrences,
				type: "song" as "song",
				viewsLog,
			};
		})
		.filter((d) => d !== null);

	// If no results, return empty array
	if (results.length === 0) return [];

	// Normalize occurrences and viewsLog
	const maxOccurrences = Math.max(...results.map((r) => r.occurrences));
	const minViewsLog = Math.min(...results.map((r) => r.viewsLog));
	const maxViewsLog = Math.max(...results.map((r) => r.viewsLog));
	const viewsLogRange = maxViewsLog - minViewsLog || 1; // Prevent division by zero

	// Calculate weighted rank (0-1 range)
	// Weight: 0.6 for occurrences, 0.4 for viewsLog
	const normalizedResults = results.map((result) => {
		const normalizedOccurrences = maxOccurrences > 0 ? result.occurrences / maxOccurrences : 0;
		const normalizedViewsLog = (result.viewsLog - minViewsLog) / viewsLogRange;

		// Weighted combination
		const rank =
			normalizedOccurrences * 0.3 + result.lengthRatio * 0.5 + normalizedViewsLog * 0.2;

		return {
			data: result.data.songs,
			rank: Math.min(Math.max(rank, 0), 1), // Ensure rank is between 0 and 1
			type: result.type,
		};
	});

	return normalizedResults;
};

const getDBVideoSearchResult = async (searchQuery: string) => {
	const aid = biliIDToAID(searchQuery);
	if (!aid) return [];
	const results = await db
		.select()
		.from(bilibiliMetadata)
		.innerJoin(latestVideoSnapshot, eq(bilibiliMetadata.aid, latestVideoSnapshot.aid))
		.where(eq(bilibiliMetadata.aid, aid));
	return results.map((video) => ({
		data: { views: video.latest_video_snapshot.views, ...video.bilibili_metadata },
		rank: 1, // Exact match
		type: "bili-video-db" as "bili-video-db",
	}));
};

const getVideoSearchResult = async (searchQuery: string) => {
	const aid = biliIDToAID(searchQuery);
	if (!aid) return [];
	let data: VideoInfoData;
	const cachedData = await retrieveVideoInfoFromCache(aid);
	if (cachedData) {
		data = cachedData;
	} else {
		const result = await getVideoInfo(aid, "getVideoInfo");
		if (typeof result === "number") return [];
		data = result.data;
		const cacheKey = `cvsa:videoInfo:av${aid}`;
		await redis.setex(cacheKey, 60, JSON.stringify(data));
	}
	return [
		{
			data: data,
			rank: 0.99, // Exact match
			type: "bili-video" as "bili-video",
		},
	];
};

const BiliVideoDataSchema = BiliVideoSchema.extend({
	views: z.number(),
});

export const searchHandler = new Elysia({ prefix: "/search" }).get(
	"/result",
	async ({ query }) => {
		const start = performance.now();
		const searchQuery = query.query;
		const [songResults, videoResults, dbVideoResults] = await Promise.all([
			getSongSearchResult(searchQuery),
			getVideoSearchResult(searchQuery),
			getDBVideoSearchResult(searchQuery),
		]);

		const combinedResults = [...songResults, ...videoResults, ...dbVideoResults];
		const data = combinedResults.sort((a, b) => b.rank - a.rank);
		const end = performance.now();
		return {
			data,
			elapsedMs: end - start,
		};
	},
	{
		detail: {
			description:
				"This endpoint performs a comprehensive search across songs and videos in the database. \
			It searches for songs by name and videos by bilibili ID (av/BV format). The results are ranked \
			by relevance using a weighted algorithm that considers search term frequency, title length, \
			and view count. Returns search results with performance timing information.",
			summary: "Search songs and videos",
		},
		query: z.object({
			query: z.string(),
		}),
		response: {
			200: z.object({
				data: z.array(
					z.union([
						z.object({
							data: SongSchema,
							rank: z.number(),
							type: z.literal("song"),
						}),
						z.object({
							data: BiliVideoDataSchema,
							rank: z.number(),
							type: z.literal("bili-video-db"),
						}),
						z.object({
							data: BiliAPIVideoMetadataSchema,
							rank: z.number(),
							type: z.literal("bili-video"),
						}),
					])
				),
				elapsedMs: z.number(),
			}),
			404: z.object({
				message: z.string(),
			}),
		},
	}
);
