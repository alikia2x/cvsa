import { Elysia, t } from "elysia";
import { db, bilibiliMetadata, eta } from "@core/drizzle";
import { eq, and, gte, lt } from "drizzle-orm";
import serverTiming from "@backend/middlewares/timing";
import z from "zod";
import { BiliVideoSchema } from "@backend/lib/schema";

type MileStoneType = "dendou" | "densetsu" | "shinwa";

const range = {
	dendou: [0, 100000, 2160],
	densetsu: [100000, 1000000, 8760],
	shinwa: [1000000, 10000000, 43800]
};

export const closeMileStoneHandler = new Elysia({ prefix: "/songs" }).use(serverTiming()).get(
	"/close-milestone/:type",
	async ({ params, timeLog }) => {
		timeLog.startTime("retrieveCandidates");
		const type = params.type;
		const min = range[type as MileStoneType][0];
		const max = range[type as MileStoneType][1];
		return db
			.select()
			.from(eta)
			.innerJoin(bilibiliMetadata, eq(bilibiliMetadata.aid, eta.aid))
			.where(
				and(
					gte(eta.currentViews, min),
					lt(eta.currentViews, max),
					lt(eta.eta, range[type as MileStoneType][2])
				)
			)
			.orderBy(eta.eta);
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
		}
	}
);
