import { Elysia } from "elysia";
import { db } from "@core/drizzle";
import { bilibiliMetadata, latestVideoSnapshot, songs } from "@core/drizzle/main/schema";
import { eq, like, or } from "drizzle-orm";
import type { BilibiliMetadataType, ProducerType, SongType } from "@core/drizzle/outerSchema";
import { BiliVideoSchema, SongSchema } from "@elysia/lib/schema";
import { z } from "zod";

interface SongSearchResult {
	type: "song";
	data: SongType;
	rank: number;
}

interface ProducerSearchResult {
	type: "producer";
	data: ProducerType;
	rank: number;
}

interface BiliVideoSearchResult {
	type: "bili-video";
	data: BilibiliMetadataType;
	rank: number; // 0 to 1
}

const getSongSearchResult = async (searchQuery: string) => {
	const data = await db
		.select()
		.from(songs)
		.innerJoin(latestVideoSnapshot, eq(songs.aid, latestVideoSnapshot.aid))
		.where(like(songs.name, `%${searchQuery}%`));

	const results = data
		.map((song) => {
			if (!song.songs.name) return null;
			const occurrences = (song.songs.name.match(new RegExp(searchQuery, "gi")) || []).length;
			const viewsLog = Math.log10(song.latest_video_snapshot.views + 1);
			return {
				type: "song" as "song",
				data: song,
				occurrences,
				viewsLog
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
		const rank = normalizedOccurrences * 0.6 + normalizedViewsLog * 0.4;

		return {
			type: result.type,
			data: result.data.songs,
			rank: Math.min(Math.max(rank, 0), 1) // Ensure rank is between 0 and 1
		};
	});

	return normalizedResults;
};

const getVideoSearchResult = async (searchQuery: string) => {
	const extractAVID = (query: string): number | null => {
		const avMatch = query.match(/av(\d+)/i);
		if (avMatch) {
			return Number.parseInt(avMatch[1]);
		}
		return 0;
	};
	const results = await db
		.select()
		.from(bilibiliMetadata)
		.innerJoin(latestVideoSnapshot, eq(bilibiliMetadata.aid, latestVideoSnapshot.aid))
		.where(
			or(
				eq(bilibiliMetadata.bvid, searchQuery),
				eq(bilibiliMetadata.aid, extractAVID(searchQuery) || 0)
			)
		);
	return results.map((video) => ({
		type: "bili-video" as "bili-video",
		data: { views: video.latest_video_snapshot.views, ...video.bilibili_metadata },
		rank: 1 // Exact match
	}));
};

const BiliVideoDataSchema = BiliVideoSchema.extend({
	views: z.number()
});

export const searchHandler = new Elysia({ prefix: "/search" }).get(
	"/result",
	async ({ query }) => {
		const start = performance.now();
		const searchQuery = query.query;
		const [songResults, videoResults] = await Promise.all([
			getSongSearchResult(searchQuery),
			getVideoSearchResult(searchQuery)
		]);

		const combinedResults = [...songResults, ...videoResults];
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
							type: z.literal("bili-video"),
							data: BiliVideoDataSchema,
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
		})
	}
);
