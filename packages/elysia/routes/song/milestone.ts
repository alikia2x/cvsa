import { Elysia, t } from "elysia";
import { dbMain } from "@core/drizzle";
import { bilibiliMetadata, latestVideoSnapshot } from "@core/drizzle/main/schema";
import { eq, and, gte, lt, desc } from "drizzle-orm";
import { getShortTermETA } from "@core/db";

type MileStoneType = "dendou" | "densetsu" | "shinwa";

const range = {
	dendou: [90000, 99999, 100000],
	densetsu: [900000, 999999, 1000000],
	shinwa: [5000000, 9999999, 10000000]
};

export const closeMileStoneHandler = new Elysia({ prefix: "/song" }).get(
	"/close-milestone/:type",
	async (c) => {
		const type = c.params.type;
		const min = range[type as MileStoneType][0];
		const max = range[type as MileStoneType][1];
		const data = await dbMain
			.select()
			.from(bilibiliMetadata)
			.innerJoin(latestVideoSnapshot, eq(latestVideoSnapshot.aid, bilibiliMetadata.aid))
			.where(and(gte(latestVideoSnapshot.views, min), lt(latestVideoSnapshot.views, max)))
			.orderBy(desc(latestVideoSnapshot.views));
		type Row = (typeof data)[number];
		type Result = Row & {
			eta: number;
		};
		const result: Result[] = [];
		for (let i = 0; i < data.length; i++) {
			const aid = data[i].bilibili_metadata.aid;
			const eta = await getShortTermETA(aid, range[type as MileStoneType][2]);
			result.push({
				...data[i],
				eta
			});
		}
		result.sort((a, b) => a.eta - b.eta);
		return result;
	},
	{
		response: {
			200: t.Array(t.Any()),
			404: t.Object({
				message: t.String()
			})
		}
	}
);
