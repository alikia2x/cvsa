import { biliIDToAID } from "@backend/lib/bilibiliID";
import { SnapshotQueue } from "@backend/lib/mq";
import { ErrorResponseSchema } from "@backend/src/schema";
import { db, videoSnapshot } from "@core/drizzle";
import { desc, eq } from "drizzle-orm";
import { Elysia } from "elysia";
import z from "zod";

export const getVideoSnapshotsHandler = new Elysia({ prefix: "/video" }).get(
	"/:id/snapshots",
	async (c) => {
		const id = c.params.id;
		const aid = biliIDToAID(id);

		if (!aid) {
			return c.status(400, {
				code: "MALFORMED_SLOT",
				message:
					"We cannot parse the video ID, or we currently do not support this format.",
				errors: [],
			});
		}

		const data = await db
			.select()
			.from(videoSnapshot)
			.where(eq(videoSnapshot.aid, aid))
			.orderBy(desc(videoSnapshot.createdAt));

		if (data.length === 0) {
			await SnapshotQueue.add("directSnapshot", {
				aid,
			});
		}

		return data;
	},
	{
		response: {
			200: z.array(
				z.object({
					id: z.number(),
					createdAt: z.string(),
					views: z.number(),
					coins: z.number().nullable(),
					likes: z.number().nullable(),
					favorites: z.number().nullable(),
					shares: z.number().nullable(),
					danmakus: z.number().nullable(),
					aid: z.number(),
					replies: z.number().nullable(),
				})
			),
			400: ErrorResponseSchema,
			500: ErrorResponseSchema,
		},
		detail: {
			summary: "Get video snapshots",
			description:
				"This endpoint retrieves historical view count snapshots for a bilibili video. It accepts video IDs in av or BV format \
			and returns a chronological list of snapshots showing how the video's statistics (views, likes, coins, favorites, etc.) \
			have changed over time. If no snapshots exist for the video, it automatically queues a snapshot job to collect initial data. \
			Results are ordered by creation date in descending order.",
		},
	}
);
