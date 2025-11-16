import { Elysia, t } from "elysia";
import { requireAuth } from "@backend/middlewares/auth";
import { songs, history, db } from "@core/drizzle";
import { eq } from "drizzle-orm";

export const deleteSongHandler = new Elysia({ prefix: "/song" }).use(requireAuth).delete(
	"/:id",
	async ({ params, user }) => {
		const id = Number(params.id);
		await db.update(songs).set({ deleted: true }).where(eq(songs.id, id));
		await db.insert(history).values({
			objectId: id,
			changeType: "del-song",
			changedBy: user!.unqId,
			data: null
		});
		return {
			message: `Successfully deleted song ${id}.`
		};
	},
	{
		response: {
			200: t.Object({
				message: t.String()
			}),
			401: t.Object({
				message: t.String()
			}),
			500: t.Object({
				message: t.String()
			})
		},
		params: t.Object({
			id: t.String()
		})
	}
);
