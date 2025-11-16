import { Elysia, t } from "elysia";
import { biliIDToAID } from "@elysia/lib/bilibiliID";
import { requireAuth } from "@elysia/middlewares/auth";
import { LatestVideosQueue } from "@elysia/lib/mq";
import { db } from "@core/drizzle";
import { songs } from "@core/drizzle/main/schema";
import { eq, and } from "drizzle-orm";

export const addSongHandler = new Elysia()
	.use(requireAuth)
	.post(
		"/song/import/bilibili",
		async ({ body, status }) => {
			const id = body.id;
			const aid = biliIDToAID(id);
			if (!aid) {
				return status(400, {
					message:
						"We cannot parse the video ID, or we currently do not support this format."
				});
			}
			const aidExistsInSongs = await db
				.select()
				.from(songs)
				.where(and(eq(songs.aid, aid), eq(songs.deleted, false)))
				.limit(1);
			if (aidExistsInSongs.length > 0) {
				return {
					jobID: -1,
					message: "Video already exists in the songs table."
				};
			}
			const job = await LatestVideosQueue.add("getVideoInfo", {
				aid: aid,
				insertSongs: true
			});
			if (!job.id) {
				return status(500, {
					message: "Failed to enqueue job to add song."
				});
			}
			return status(201, {
				message: "Successfully created import session.",
				jobID: job.id
			});
		},
		{
			response: {
				201: t.Object({
					message: t.String(),
					jobID: t.String()
				}),
				400: t.Object({
					message: t.String()
				}),
				401: t.Object({
					message: t.String()
				}),
				500: t.Object({
					message: t.String()
				})
			},
			body: t.Object({
				id: t.String()
			})
		}
	)
	.get(
		"/song/import/:id/status",
		async ({ params, status }) => {
			const jobID = params.id;
			if (parseInt(jobID) === -1) {
				return {
					id: jobID,
					state: "completed",
					result: {
						message: "Video already exists in the songs table."
					}
				}
			}
			const job = await LatestVideosQueue.getJob(jobID);
			if (!job) {
				return status(404, {
					message: "Job not found."
				});
			}
			const state = await job.getState();
			return {
				id: job.id!,
				state,
				result: job.returnvalue,
				failedReason: job.failedReason
			};
		},
		{
			response: {
				200: t.Object({
					id: t.String(),
					state: t.String(),
					result: t.Optional(t.Any()),
					failedReason: t.Optional(t.String())
				}),
				404: t.Object({
					message: t.String()
				})
			},
			params: t.Object({
				id: t.String()
			})
		}
	);
