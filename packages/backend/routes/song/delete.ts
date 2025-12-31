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
			changedBy: user!.unqId,
			changeType: "del-song",
			data: null,
			objectId: id,
		});
		return {
			message: `Successfully deleted song ${id}.`,
		};
	},
	{
		detail: {
			description:
				"This endpoint allows authenticated users to soft-delete a song from the database. \
			The song is marked as deleted rather than being permanently removed, preserving data integrity. \
			The deletion is logged in the history table for audit purposes. Requires authentication and appropriate permissions.",
			summary: "Delete song",
		},
		params: t.Object({
			id: t.String(),
		}),
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
	}
);
