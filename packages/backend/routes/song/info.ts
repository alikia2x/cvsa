import { Elysia, t } from "elysia";
import { dbMain } from "@core/drizzle";
import { relations, singer, songs } from "@core/drizzle/main/schema";
import { eq, and } from "drizzle-orm";
import { bv2av } from "@elysia/lib/bilibiliID";
import { requireAuth } from "@elysia/middlewares/auth";

async function getSongIDFromBiliID(id: string) {
	let aid: number;
	if (id.startsWith("BV1")) {
		aid = bv2av(id as `BV1${string}`);
	} else if (id.startsWith("av")) {
		aid = Number.parseInt(id.slice(2));
	} else {
		return null;
	}
	const songID = await dbMain
		.select({ id: songs.id })
		.from(songs)
		.where(eq(songs.aid, aid))
		.limit(1);
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
	const songInfo = await dbMain
		.select()
		.from(songs)
		.where(and(eq(songs.id, id), eq(songs.deleted, false)))
		.limit(1);
	return songInfo[0];
}

async function getSingers(id: number) {
	const singers = await dbMain
		.select({
			singers: singer.name
		})
		.from(relations)
		.innerJoin(singer, eq(relations.targetId, singer.id))
		.where(
			and(
				eq(relations.sourceId, id),
				eq(relations.sourceType, "song"),
				eq(relations.relation, "sing")
			)
		);
	return singers.map((singer) => singer.singers);
}

const songInfoGetHandler = new Elysia({ prefix: "/song" }).get(
	"/:id/info",
	async ({ params, status }) => {
		const id = params.id;
		const songID = await getSongID(id);
		if (!songID) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found."
			});
		}
		const info = await getSongInfo(songID);
		if (!info) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found."
			});
		}
		const singers = await getSingers(info.id);
		return {
			id: info.id,
			name: info.name,
			aid: info.aid,
			producer: info.producer,
			duration: info.duration,
			singers: singers,
			cover: info.image || undefined,
			publishedAt: info.publishedAt
		};
	},
	{
		response: {
			200: t.Object({
				id: t.Number(),
				name: t.Union([t.String(), t.Null()]),
				aid: t.Union([t.Number(), t.Null()]),
				producer: t.Union([t.String(), t.Null()]),
				duration: t.Union([t.Number(), t.Null()]),
				singers: t.Array(t.String()),
				cover: t.Optional(t.String()),
				publishedAt: t.Union([t.String(), t.Null()]),
			}),
			404: t.Object({
				code: t.String(),
				message: t.String()
			})
		},
		headers: t.Object({
			Authorization: t.Optional(t.String())
		}),
		detail: {
			summary: "Get information of a song",
			description:
				"This endpoint retrieves detailed information about a song using its unique ID, \
			which can be provided in several formats. \
			The endpoint accepts a song ID in either a numerical format as the internal ID in our database\
			 or as a bilibili video ID (either av or BV format). \
			 It responds with the song's name, bilibili ID (av), producer, duration, and associated singers."
		}
	}
);

const songInfoUpdateHandler = new Elysia({ prefix: "/song" }).use(requireAuth).patch(
	"/:id/info",
	async ({ params, status, body }) => {
		const id = params.id;
		const songID = await getSongID(id);
		if (!songID) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found."
			});
		}
		const info = await getSongInfo(songID);
		if (!info) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found."
			});
		}

		if (body.name) {
			await dbMain.update(songs).set({ name: body.name }).where(eq(songs.id, songID));
		}
		if (body.producer) {
			await dbMain
				.update(songs)
				.set({ producer: body.producer })
				.where(eq(songs.id, songID))
				.returning();
		}
		const updatedData = await dbMain.select().from(songs).where(eq(songs.id, songID));
		return {
			message: "Successfully updated song info.",
			updated: updatedData.length > 0 ? updatedData[0] : null
		};
	},
	{
		response: {
			200: t.Object({
				message: t.String(),
				updated: t.Any()
			}),
			401: t.Object({
				message: t.String()
			}),
			404: t.Object({
				message: t.String(),
				code: t.String()
			})
		},
		body: t.Object({
			name: t.Optional(t.String()),
			producer: t.Optional(t.String())
		})
	}
);

export const songInfoHandler = new Elysia().use(songInfoGetHandler).use(songInfoUpdateHandler);
