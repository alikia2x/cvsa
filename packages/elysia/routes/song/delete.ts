import { Elysia, t } from "elysia";
import { requireAuth } from "@elysia/middlewares/auth";
import { db } from "@core/drizzle";
import { songs } from "@core/drizzle/main/schema";
import { eq } from "drizzle-orm";

export const deleteSongHandler = new Elysia({ prefix: "/song" }).use(requireAuth).delete(
	"/:id",
	async ({ params }) => {
		const id = Number(params.id);
		await db.update(songs).set({ deleted: true }).where(eq(songs.id, id));
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
