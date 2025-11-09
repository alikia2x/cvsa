import { Elysia, t } from "elysia";
import { biliIDToAID } from "@elysia/lib/bilibiliID";
import { requireAuth } from "@elysia/middlewares/auth";
import { LatestVideosQueue } from "@elysia/lib/mq";

export const addSongHandler = new Elysia()
	.use(requireAuth)
	.post(
		"/song/import/bilibili",
		async ({ body, status }) => {
			const id = body.id;
			const aid = biliIDToAID(id);
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
