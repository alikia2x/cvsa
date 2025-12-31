import { biliIDToAID } from "@backend/lib/bilibiliID";
import { LatestVideosQueue } from "@backend/lib/mq";
import { requireAuth } from "@backend/middlewares/auth";
import { db, songs } from "@core/drizzle";
import { and, eq } from "drizzle-orm";
import { Elysia, t } from "elysia";

export const addSongHandler = new Elysia()
	.use(requireAuth)
	.post(
		"/song/import/bilibili",
		async ({ body, status, user }) => {
			const id = body.id;
			const aid = biliIDToAID(id);
			if (!aid) {
				return status(400, {
					message:
						"We cannot parse the video ID, or we currently do not support this format.",
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
					message: "Video already exists in the songs table.",
				};
			}
			const job = await LatestVideosQueue.add("getVideoInfo", {
				aid: aid,
				insertSongs: true,
				uid: user!.unqId,
			});
			if (!job.id) {
				return status(500, {
					message: "Failed to enqueue job to add song.",
				});
			}
			return status(201, {
				jobID: job.id,
				message: "Successfully created import session.",
			});
		},
		{
			body: t.Object({
				id: t.String(),
			}),
			detail: {
				description:
					"This endpoint allows authenticated users to import a song from bilibili by providing a video ID. \
				The video ID can be in av or BV format. The system validates the ID format, checks if the video already \
				exists in the database, and if not, creates a background job to fetch video metadata and add it to the songs collection. \
				Returns the job ID for tracking the import progress.",
				summary: "Import song from bilibili",
			},
			response: {
				201: t.Object({
					jobID: t.String(),
					message: t.String(),
				}),
				400: t.Object({
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
	)
	.get(
		"/song/import/:id/status",
		async ({ params, status }) => {
			const jobID = params.id;
			if (parseInt(jobID) === -1) {
				return {
					id: jobID,
					result: {
						message: "Video already exists in the songs table.",
					},
					state: "completed",
				};
			}
			const job = await LatestVideosQueue.getJob(jobID);
			if (!job) {
				return status(404, {
					message: "Job not found.",
				});
			}
			const state = await job.getState();
			return {
				failedReason: job.failedReason,
				id: job.id!,
				result: job.returnvalue,
				state,
			};
		},
		{
			detail: {
				description:
					"This endpoint retrieves the current status of a song import job. It returns the job state \
				(completed, failed, active, etc.), the result if completed, and any failure reason if the job failed. \
				Use this endpoint to monitor the progress of song imports initiated through the import endpoint.",
				summary: "Check import job status",
			},
			params: t.Object({
				id: t.String(),
			}),
			response: {
				200: t.Object({
					failedReason: t.Optional(t.String()),
					id: t.String(),
					result: t.Optional(t.Any()),
					state: t.String(),
				}),
				404: t.Object({
					message: t.String(),
				}),
			},
		}
	);
