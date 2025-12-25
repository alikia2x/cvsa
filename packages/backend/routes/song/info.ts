import { bv2av } from "@backend/lib/bilibiliID";
import { requireAuth } from "@backend/middlewares/auth";
import { db, eta, history, songs, videoSnapshot } from "@core/drizzle";
import { and, desc, eq } from "drizzle-orm";
import { Elysia, t } from "elysia";

async function getSongIDFromBiliID(id: string) {
	let aid: number;
	if (id.startsWith("BV1")) {
		aid = bv2av(id as `BV1${string}`);
	} else if (id.startsWith("av")) {
		aid = Number.parseInt(id.slice(2));
	} else {
		return null;
	}
	const songID = await db.select({ id: songs.id }).from(songs).where(eq(songs.aid, aid)).limit(1);
	if (songID.length > 0) {
		return songID[0].id;
	}
	return null;
}

async function getSongID(id: string) {
	let songID: number | null = null;
	if (id.startsWith("BV1") || id.startsWith("av")) {
		const r = await getSongIDFromBiliID(id);
		if (r) songID = r;
	}
	if (!songID) {
		songID = Number.parseInt(id);
	}
	return songID;
}

async function getSongInfo(id: number) {
	const songInfo = await db
		.select()
		.from(songs)
		.where(and(eq(songs.id, id), eq(songs.deleted, false)))
		.limit(1);
	return songInfo[0];
}

export const songHandler = new Elysia({ prefix: "/song/:id" })
	.resolve(async ({ params, status }) => {
		const id = params.id;
		const songID = await getSongID(id);
		if (Number.isNaN(songID)) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found.",
			});
		}
		return {
			songID,
		};
	})
	.get(
		"/info",
		async ({ status, songID }) => {
			const info = await getSongInfo(songID);
			if (!info) {
				return status(404, {
					code: "SONG_NOT_FOUND",
					message: "Given song cannot be found.",
				});
			}
			return {
				aid: info.aid,
				cover: info.image || undefined,
				duration: info.duration,
				id: info.id,
				name: info.name,
				producer: info.producer,
				publishedAt: info.publishedAt,
			};
		},
		{
			detail: {
				description:
					"This endpoint retrieves detailed information about a song using its unique ID, \
			which can be provided in several formats. \
			The endpoint accepts a song ID in either a numerical format as the internal ID in our database\
			 or as a bilibili video ID (either av or BV format). \
			 It responds with the song's name, bilibili ID (av), producer, duration, and associated singers.",
				summary: "Get information of a song",
			},
			headers: t.Object({
				Authorization: t.Optional(t.String()),
			}),
			response: {
				200: t.Object({
					aid: t.Union([t.Number(), t.Null()]),
					cover: t.Optional(t.String()),
					duration: t.Union([t.Number(), t.Null()]),
					id: t.Number(),
					name: t.Union([t.String(), t.Null()]),
					producer: t.Union([t.String(), t.Null()]),
					publishedAt: t.Union([t.String(), t.Null()]),
				}),
				404: t.Object({
					code: t.String(),
					message: t.String(),
				}),
			},
		}
	)
	.get("/snapshots", async ({ status, songID }) => {
		const r = await db.select().from(songs).where(eq(songs.id, songID)).limit(1);
		if (r.length === 0) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found.",
			});
		}
		const song = r[0];
		const aid = song.aid;
		if (!aid) {
			return status(404, {
				message: "Given song is not associated with any bilibili video.",
			});
		}
		return db
			.select()
			.from(videoSnapshot)
			.where(eq(videoSnapshot.aid, aid))
			.orderBy(desc(videoSnapshot.createdAt));
	})
	.get("/eta", async ({ status, songID }) => {
		const r = await db.select().from(songs).where(eq(songs.id, songID)).limit(1);
		if (r.length === 0) {
			return status(404, {
				code: "SONG_NOT_FOUND",
				message: "Given song cannot be found.",
			});
		}
		const song = r[0];
		const aid = song.aid;
		if (!aid) {
			return status(404, {
				message: "Given song is not associated with any bilibili video.",
			});
		}
		const data = await db.select().from(eta).where(eq(eta.aid, aid));
		return {
			aid: data[0].aid,
			eta: data[0].eta,
			speed: data[0].speed,
			updatedAt: data[0].updatedAt,
			views: data[0].currentViews,
		};
	})
	.use(requireAuth)
	.patch(
		"/info",
		async ({ status, body, user, songID }) => {
			const info = await getSongInfo(songID);
			if (!info) {
				return status(404, {
					code: "SONG_NOT_FOUND",
					message: "Given song cannot be found.",
				});
			}

			if (body.name) {
				await db.update(songs).set({ name: body.name }).where(eq(songs.id, songID));
			}
			if (body.producer) {
				await db
					.update(songs)
					.set({ producer: body.producer })
					.where(eq(songs.id, songID))
					.returning();
			}
			const updatedData = await db.select().from(songs).where(eq(songs.id, songID));
			await db.insert(history).values({
				changedBy: user!.unqId,
				changeType: "update-song",
				data:
					updatedData.length > 0
						? {
								new: updatedData[0],
								old: info,
							}
						: null,
				objectId: songID,
			});
			return {
				message: "Successfully updated song info.",
				updated: updatedData.length > 0 ? updatedData[0] : null,
			};
		},
		{
			body: t.Object({
				name: t.Optional(t.String()),
				producer: t.Optional(t.String()),
			}),
			detail: {
				description:
					"This endpoint allows authenticated users to update song metadata. It accepts partial updates \
			for song name and producer fields. The endpoint validates the song ID (accepting both internal database IDs \
			and bilibili video IDs in av/BV format), applies the requested changes, and logs the update in the history table \
			for audit purposes. Requires authentication.",
				summary: "Update song information",
			},
			response: {
				200: t.Object({
					message: t.String(),
					updated: t.Any(),
				}),
				401: t.Object({
					message: t.String(),
				}),
				404: t.Object({
					code: t.String(),
					message: t.String(),
				}),
			},
		}
	);
