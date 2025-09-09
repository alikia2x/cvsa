import { dbMain } from "~/drizzle";
import { bilibiliMetadata } from "~db/main/schema";
import { eq } from "drizzle-orm";

export const getVideoAID = async (id: string) => {
	"use server";
	if (id.startsWith("av")) {
		return parseInt(id.slice(2));
	} else if (id.startsWith("BV")) {
		const data = await dbMain.select().from(bilibiliMetadata).where(eq(bilibiliMetadata.bvid, id));
		return data[0].aid;
	} else {
		return null;
	}
};