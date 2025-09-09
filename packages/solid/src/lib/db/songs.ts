import { dbMain } from "~/drizzle";
import { songs } from "~db/main/schema";
import { eq } from "drizzle-orm";

export const findSongIDFromAID = async (aid: number) => {
	"use server";
	const data = await dbMain
		.select({
			id: songs.id
		})
		.from(songs)
		.where(eq(songs.aid, aid))
		.limit(1);
	return data[0].id;
};