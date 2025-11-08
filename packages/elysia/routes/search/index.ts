import { Elysia, t } from "elysia";
import { db } from "@core/drizzle";
import { bilibiliMetadata, latestVideoSnapshot, songs } from "@core/drizzle/main/schema";
import { eq, like, or } from "drizzle-orm";
import type { BilibiliMetadataType, ProducerType, SongType } from "@core/drizzle/outerSchema";

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
		.where(like(songs.name, `%${searchQuery}%`))
		.limit(10);

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
	const results = await db
		.select()
		.from(bilibiliMetadata)
		.where(
			or(
				eq(bilibiliMetadata.bvid, searchQuery),
				eq(bilibiliMetadata.aid, Number.parseInt(searchQuery) || 0)
			)
		)
		.limit(10);
	return results.map((video) => ({
		type: "bili-video" as "bili-video",
		data: video,
		rank: 1 // Exact match
	}));
};

const BiliVideoDataSchema = t.Object({
	duration: t.Union([t.Number(), t.Null()]),
	id: t.Number(),
	aid: t.Number(),
	publishedAt: t.Union([t.String(), t.Null()]),
	createdAt: t.Union([t.String(), t.Null()]),
	description: t.Union([t.String(), t.Null()]),
	bvid: t.Union([t.String(), t.Null()]),
	uid: t.Union([t.Number(), t.Null()]),
	tags: t.Union([t.String(), t.Null()]),
	title: t.Union([t.String(), t.Null()]),
	status: t.Number(),
	coverUrl: t.Union([t.String(), t.Null()])
});

const SongDataSchema = t.Object({
	duration: t.Union([t.Number(), t.Null()]),
	name: t.Union([t.String(), t.Null()]),
	id: t.Number(),
	aid: t.Union([t.Number(), t.Null()]),
	publishedAt: t.Union([t.String(), t.Null()]),
	type: t.Union([t.Number(), t.Null()]),
	neteaseId: t.Union([t.Number(), t.Null()]),
	createdAt: t.String(),
	updatedAt: t.String(),
	deleted: t.Boolean(),
	image: t.Union([t.String(), t.Null()]),
	producer: t.Union([t.String(), t.Null()])
});

export const searchHandler = new Elysia({ prefix: "/search" }).get(
	"/result",
	async ({ query }) => {
		const searchQuery = query.query;
		const [songResults, videoResults] = await Promise.all([
			getSongSearchResult(searchQuery),
			getVideoSearchResult(searchQuery)
		]);

		const combinedResults: (SongSearchResult | BiliVideoSearchResult)[] = [
			...songResults,
			...videoResults
		];
		return combinedResults.sort((a, b) => b.rank - a.rank);
	},
	{
		response: {
			200: t.Array(
				t.Union([
					t.Object({
						type: t.Literal("song"),
						data: SongDataSchema,
						rank: t.Number()
					}),
					t.Object({
						type: t.Literal("bili-video"),
						data: BiliVideoDataSchema,
						rank: t.Number()
					})
				])
			),
			404: t.Object({
				message: t.String()
			})
		},
		query: t.Object({
			query: t.String()
		})
	}
);
