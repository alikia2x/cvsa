import { biliIDToAID } from "@backend/lib/bilibiliID";
import { BiliVideoSchema } from "@backend/lib/schema";
import requireAuth from "@backend/middlewares/auth";
import { ErrorResponseSchema } from "@backend/src/schema";
import { bilibiliMetadata, db, videoTypeLabelInInternal } from "@core/drizzle";
import { eq, sql } from "drizzle-orm";
import { Elysia, t } from "elysia";
import z from "zod";

const videoSchema = BiliVideoSchema.omit({ publishedAt: true })
	.omit({ createdAt: true })
	.omit({ coverUrl: true })
	.extend({
		views: z.number(),
		username: z.string(),
		uid: z.number(),
		published_at: z.string(),
		createdAt: z.string(),
		cover_url: z.string(),
	});

export const getUnlabelledVideos = new Elysia({ prefix: "/videos" }).use(requireAuth).get(
	"/unlabelled",
	async ({ user }) => {
		return db.execute<z.infer<typeof videoSchema>>(sql`
            SELECT bm.*, ls.views, bu.username, bu.uid
			FROM (
				SELECT *
				FROM bilibili_metadata
				TABLESAMPLE SYSTEM (0.1)
				ORDER BY RANDOM()
				LIMIT 1
			) bm
			JOIN latest_video_snapshot ls
				ON ls.aid = bm.aid
			JOIN bilibili_user bu
				ON bu.uid = bm.uid
            UNION
			SELECT bm.*, ls.views, bu.username, bu.uid
			FROM (
				 SELECT *
				 FROM bilibili_metadata
				 WHERE aid IN (
					 SELECT aid
					 FROM internal.video_type_label
					 TABLESAMPLE SYSTEM (8)
					 WHERE video_type_label."user" != ${user!.unqId}
					 AND video_type_label."user" = 'bvZMWcgYL2dr6vsJ' 
					 ORDER BY RANDOM()
					 LIMIT 19
				 )
			 ) bm
				 JOIN latest_video_snapshot ls
					  ON ls.aid = bm.aid
				 JOIN bilibili_user bu
					  ON bu.uid = bm.uid
		`);
	},
	{
		response: {
			200: z.array(videoSchema),
			400: ErrorResponseSchema,
			500: ErrorResponseSchema,
		},
	}
);

export const postVideoLabel = new Elysia({ prefix: "/video" }).use(requireAuth).post(
	"/:id/label",
	async ({ params, body, status, user }) => {
		const id = params.id;
		const aid = biliIDToAID(id);
		const label = body.label;

		if (!aid) {
			return status(400, {
				code: "MALFORMED_SLOT",
				message:
					"We cannot parse the video ID, or we currently do not support this format.",
				errors: [],
			});
		}

		const video = await db
			.select()
			.from(bilibiliMetadata)
			.where(eq(bilibiliMetadata.aid, aid))
			.limit(1);

		if (video.length === 0) {
			return status(400, {
				code: "VIDEO_NOT_FOUND",
				message: "Video not found",
				errors: [],
			});
		}

		await db.insert(videoTypeLabelInInternal).values({
			aid,
			label,
			user: user!.unqId,
		});

		return status(201, {
			message: `Labelled video av${aid} as ${label}`,
		});
	},
	{
		body: t.Object({
			label: t.Boolean(),
		}),
	}
);
