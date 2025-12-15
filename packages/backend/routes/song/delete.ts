import { requireAuth } from "@backend/middlewares/auth";
import { db, history, songs } from "@core/drizzle";
import { eq } from "drizzle-orm";
import { Elysia, t } from "elysia";

export const deleteSongHandler = new Elysia({ prefix: "/song" }).use(requireAuth).delete(
	"/:id",
	async ({ params, user }) => {
		const id = Number(params.id);
		await db.update(songs).set({ deleted: true }).where(eq(songs.id, id));
		await db.insert(history).values({
			objectId: id,
			changeType: "del-song",
			changedBy: user!.unqId,
			data: null,
		});
		return {
			message: `Successfully deleted song ${id}.`,
		};
	},
	{
		response: {
			200: t.Object({
				message: t.String(),
			}),
			401: t.Object({
				message: t.String(),
			}),
			500: t.Object({
				message: t.String(),
			}),
		},
		params: t.Object({
			id: t.String(),
		}),
		detail: {
			summary: "Delete song",
			description:
				"This endpoint allows authenticated users to soft-delete a song from the database. \
			The song is marked as deleted rather than being permanently removed, preserving data integrity. \
			The deletion is logged in the history table for audit purposes. Requires authentication and appropriate permissions.",
		},
	}
);
