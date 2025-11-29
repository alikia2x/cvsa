import { Elysia } from "elysia";
import { db, bilibiliMetadata, latestVideoSnapshot, songs } from "@core/drizzle";
import { eq, ilike } from "drizzle-orm";
import { BiliAPIVideoMetadataSchema, BiliVideoSchema, SongSchema } from "@backend/lib/schema";
import { z } from "zod";
import { getVideoInfo } from "@core/net/getVideoInfo";
import { biliIDToAID } from "@backend/lib/bilibiliID";
import { retrieveVideoInfoFromCache } from "../video/metadata";
import { redis } from "@core/db/redis";

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
				type: "song" as "song",
				data: song,
				occurrences,
				viewsLog,
				lengthRatio
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
			type: result.type,
			data: result.data.songs,
			rank: Math.min(Math.max(rank, 0), 1) // Ensure rank is between 0 and 1
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
		type: "bili-video-db" as "bili-video-db",
		data: { views: video.latest_video_snapshot.views, ...video.bilibili_metadata },
		rank: 1 // Exact match
	}));
};

const getVideoSearchResult = async (searchQuery: string) => {
	const aid = biliIDToAID(searchQuery);
	if (!aid) return [];
	let data;
	const cachedData = await retrieveVideoInfoFromCache(aid);
	if (cachedData) {
		data = cachedData;
	} else {
		data = await getVideoInfo(aid, "getVideoInfo");
		if (typeof data === "number") return [];
		const cacheKey = `cvsa:videoInfo:av${aid}`;
		await redis.setex(cacheKey, 60, JSON.stringify(data));
	}
	return [
		{
			type: "bili-video" as "bili-video",
			data: data.data,
			rank: 0.99 // Exact match
		}
	];
};

const BiliVideoDataSchema = BiliVideoSchema.extend({
	views: z.number()
});

export const searchHandler = new Elysia({ prefix: "/search" }).get(
	"/result",
	async ({ query }) => {
		const start = performance.now();
		const searchQuery = query.query;
		const [songResults, videoResults, dbVideoResults] = await Promise.all([
			getSongSearchResult(searchQuery),
			getVideoSearchResult(searchQuery),
			getDBVideoSearchResult(searchQuery)
		]);

		const combinedResults = [...songResults, ...videoResults, ...dbVideoResults];
		const data = combinedResults.sort((a, b) => b.rank - a.rank);
		const end = performance.now();
		return {
			data,
			elapsedMs: end - start
		};
	},
	{
		response: {
			200: z.object({
				elapsedMs: z.number(),
				data: z.array(
					z.union([
						z.object({
							type: z.literal("song"),
							data: SongSchema,
							rank: z.number()
						}),
						z.object({
							type: z.literal("bili-video-db"),
							data: BiliVideoDataSchema,
							rank: z.number()
						}),
						z.object({
							type: z.literal("bili-video"),
							data: BiliAPIVideoMetadataSchema,
							rank: z.number()
						})
					])
				)
			}),
			404: z.object({
				message: z.string()
			})
		},
		query: z.object({
			query: z.string()
		}),
		detail: {
			summary: "Search songs and videos",
			description:
				"This endpoint performs a comprehensive search across songs and videos in the database. \
			It searches for songs by name and videos by bilibili ID (av/BV format). The results are ranked \
			by relevance using a weighted algorithm that considers search term frequency, title length, \
			and view count. Returns search results with performance timing information."
		}
	}
);
