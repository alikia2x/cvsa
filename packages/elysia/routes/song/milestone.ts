import { Elysia, t } from "elysia";
import { dbMain } from "@core/drizzle";
import { bilibiliMetadata, latestVideoSnapshot } from "@core/drizzle/main/schema";
import { eq, and, gte, lt, desc } from "drizzle-orm";

type MileStoneType = "dendou" | "densetsu" | "shinwa";

const range = {
	dendou: [90000, 99999],
	densetsu: [900000, 999999],
	shinwa: [5000000, 9999999]
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
        const aids = data.map((song) => song.bilibili_metadata.aid);
        for (const aid of aids) {
            
        }

		return data;
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
