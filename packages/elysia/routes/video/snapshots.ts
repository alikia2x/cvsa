import { Elysia } from "elysia";
import { dbMain } from "@core/drizzle";
import { videoSnapshot } from "@core/drizzle/main/schema";
import { bv2av } from "@elysia/lib/av_bv";
import { ErrorResponseSchema } from "@elysia/src/schema";
import { eq, desc } from "drizzle-orm";
import z from "zod";

export const getVideoSnapshotsHandler = new Elysia({ prefix: "/video" }).get(
	"/:id/snapshots",
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

		const data = await dbMain
			.select()
			.from(videoSnapshot)
			.where(eq(videoSnapshot.aid, aid))
			.orderBy(desc(videoSnapshot.createdAt));

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
					replies: z.number().nullable()
				})
			),
			400: ErrorResponseSchema,
			500: ErrorResponseSchema
		}
	}
);
