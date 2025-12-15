import { biliIDToAID } from "@backend/lib/bilibiliID";
import { db, eta } from "@core/drizzle";
import { eq } from "drizzle-orm";
import { Elysia, t } from "elysia";

export const songEtaHandler = new Elysia({ prefix: "/video" }).get(
	"/:id/eta",
	async ({ params, status }) => {
		const id = params.id;
		const aid = biliIDToAID(id);
		if (!aid) {
			return status(400, {
				code: "MALFORMED_SLOT",
				message:
					"We cannot parse the video ID, or we currently do not support this format.",
			});
		}
		const data = await db.select().from(eta).where(eq(eta.aid, aid));
		if (data.length === 0) {
			return status(404, {
				code: "VIDEO_NOT_FOUND",
				message: "Video not found.",
			});
		}
		return {
			aid: data[0].aid,
			eta: data[0].eta,
			views: data[0].currentViews,
			speed: data[0].speed,
			updatedAt: data[0].updatedAt,
		};
	},
	{
		response: {
			200: t.Object({
				aid: t.Number(),
				eta: t.Number(),
				views: t.Number(),
				speed: t.Number(),
				updatedAt: t.String(),
			}),
			400: t.Object({
				code: t.String(),
				message: t.String(),
			}),
			404: t.Object({
				code: t.String(),
				message: t.String(),
			}),
		},
		headers: t.Object({
			Authorization: t.Optional(t.String()),
		}),
		detail: {
			summary: "Get video milestone ETA",
			description:
				"This endpoint retrieves the estimated time to reach the next milestone for a given video. \
			It accepts video IDs in av or BV format and returns the current view count, estimated time to \
			reach the next milestone (in hours), view growth speed, and last update timestamp. Useful for \
			tracking video growth and milestone predictions.",
		},
	}
);
