import { db, latestVideoSnapshot } from "@core/drizzle";
import { eq } from "drizzle-orm";

export const getLatestSnapshot = async (aid: number) => {
	const result = await db
		.select()
		.from(latestVideoSnapshot)
		.where(eq(latestVideoSnapshot.aid, aid))
		.limit(1);
	if (result.length === 0) {
		return null;
	}
	return result[0];
};
