import { Elysia, t } from "elysia";
import { dbMain } from "@core/drizzle";
import { relations, singer, songs } from "@core/drizzle/main/schema";
import { eq, and } from "drizzle-orm";
import { bv2av } from "@elysia/lib/av_bv";

async function getSongIDFromBiliID(id: string) {
	let aid: number;
	if (id.startsWith("BV1")) {
		aid = bv2av(id as `BV1${string}`);
	} else if (id.startsWith("av")) {
		aid = Number.parseInt(id.slice(2));
	} else {
		return null;
	}
	const songID = await dbMain.select({ id: songs.id }).from(songs).where(eq(songs.aid, aid)).limit(1);
	if (songID.length > 0) {
		return songID[0].id;
	}
	return null;
}

async function getSongID(id: string) {
	let songID: number | null = null;
	if (id.startsWith("BV1") || id.startsWith("av")) {
		const r = await getSongIDFromBiliID(id);
		r && (songID = r);
	}
	if (!songID) {
		songID = Number.parseInt(id);
	}
	return songID;
}

async function getSongInfo(id: number) {
	const songInfo = await dbMain.select().from(songs).where(eq(songs.id, id)).limit(1);
	return songInfo[0];
}

async function getSingers(id: number) {
	const singers = await dbMain
		.select({
			singers: singer.name
		})
		.from(relations)
		.innerJoin(singer, eq(relations.targetId, singer.id))
		.where(and(eq(relations.sourceId, id), eq(relations.sourceType, "song"), eq(relations.relation, "sing")));
	return singers.map((singer) => singer.singers);
}

export const getSongInfoHandler = new Elysia({ prefix: "/song" }).get(
	"/:id/info",
	async (c) => {
		const id = c.params.id;
		const songID = await getSongID(id);
		if (!songID) {
			return c.status(404, {
				message: "song not found"
			});
		}
		const info = await getSongInfo(songID);
		if (!info) {
			return c.status(404, {
				message: "song not found"
			});
		}
		const singers = await getSingers(info.id);
		return {
			name: info.name,
			aid: info.aid,
			producer: info.producer,
			duration: info.duration,
			singers: singers
		};
	},
	{
		response: {
			200: t.Object({
				name: t.Union([t.String(), t.Null()]),
				aid: t.Union([t.Number(), t.Null()]),
				producer: t.Union([t.String(), t.Null()]),
				duration: t.Union([t.Number(), t.Null()]),
				singers: t.Array(t.String())
			}),
			404: t.Object({
				message: t.String()
			})
		},
		detail: {
			summary: "Get information of a song",
			description: ""
		}
	}
);
