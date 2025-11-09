import { Elysia, t } from "elysia";
import { dbMain } from "@core/drizzle";
import { relations, singer, songs } from "@core/drizzle/main/schema";
import { eq, and } from "drizzle-orm";
import { biliIDToAID, bv2av } from "@elysia/lib/bilibiliID";
import { requireAuth } from "@elysia/middlewares/auth";
import { LatestVideosQueue } from "@elysia/lib/mq";

const addSongHandler = new Elysia()
	.use(requireAuth)
	.post(
		"/song/bilibili",
		async ({ params, status, body, user }) => {
			const id = body.id;
			const aid = biliIDToAID(id);
            const job = LatestVideosQueue.add("getVideoInfo", {
                aid: aid
            })
			return {
				message: "Successfully updated song info.",
			};
		},
		{
			response: {
				200: t.Object({
					message: t.String(),
					updated: t.Any()
				}),
				401: t.Object({
					message: t.String()
				}),
				404: t.Object({
					message: t.String(),
					code: t.String()
				})
			},
			body: t.Object({
				id: t.String()
			})
		}
	);
