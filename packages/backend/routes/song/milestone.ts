import { Elysia, t } from "elysia";
import { db, bilibiliMetadata, eta } from "@core/drizzle";
import { eq, and, gte, lt } from "drizzle-orm";
import serverTiming from "@backend/middlewares/timing";
import z from "zod";
import { BiliVideoSchema } from "@backend/lib/schema";

type MileStoneType = "dendou" | "densetsu" | "shinwa";

const range = {
	dendou: [0, 100000],
	densetsu: [100000, 1000000],
	shinwa: [1000000, 10000000]
};

export const closeMileStoneHandler = new Elysia({ prefix: "/songs" }).use(serverTiming()).get(
	"/close-milestone/:type",
	async ({ params }) => {
		const type = params.type;
		const offset = params.offset;
		const limit = params.limit;
		const min = range[type as MileStoneType][0];
		const max = range[type as MileStoneType][1];
		const query = db
			.select()
			.from(eta)
			.innerJoin(bilibiliMetadata, eq(bilibiliMetadata.aid, eta.aid))
			.where(and(gte(eta.currentViews, min), lt(eta.currentViews, max)))
			.orderBy(eta.eta)
			.$dynamic();
		return query.limit(limit || 20).offset(offset || 0);
	},
	{
		response: {
			200: z.array(
				z.object({
					eta: z.object({
						aid: z.number(),
						eta: z.number(),
						speed: z.number(),
						currentViews: z.number(),
						updatedAt: z.string()
					}),
					bilibili_metadata: BiliVideoSchema
				})
			),
			404: t.Object({
				message: t.String()
			})
		},
		params: t.Object({
			type: t.String({ enum: ["dendou", "densetsu", "shinwa"] }),
			offset: t.Optional(t.Number()),
			limit: t.Optional(t.Number())
		}),
		detail: {
			summary: "Get songs close to milestones",
			description:
				"This endpoint retrieves songs that are approaching significant view count milestones. \
			It supports three milestone types: 'dendou' (0-100k views), 'densetsu' (100k-1M views), and 'shinwa' (1M-10M views). \
			For each type, it returns videos that are within the specified view range and have an estimated time to reach \
			the next milestone below the threshold. Results are ordered by estimated time to milestone."
		}
	}
);
