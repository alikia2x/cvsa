import { Elysia, t } from "elysia";
import { dbMain } from "@core/drizzle";
import { bilibiliMetadata, eta, latestVideoSnapshot } from "@core/drizzle/main/schema";
import { eq, and, gte, lt, desc } from "drizzle-orm";
import serverTiming from "@elysia/middlewares/timing";
import z from "zod";
import { BiliVideoSchema } from "@elysia/lib/schema";

type MileStoneType = "dendou" | "densetsu" | "shinwa";

const range = {
	dendou: [90000, 99999, 2160],
	densetsu: [900000, 999999, 8760],
	shinwa: [5000000, 9999999, 87600]
};

export const closeMileStoneHandler = new Elysia({ prefix: "/songs" }).use(serverTiming()).get(
	"/close-milestone/:type",
	async ({ params, timeLog }) => {
		timeLog.startTime("retrieveCandidates");
		const type = params.type;
		const min = range[type as MileStoneType][0];
		const max = range[type as MileStoneType][1];
		return dbMain
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
