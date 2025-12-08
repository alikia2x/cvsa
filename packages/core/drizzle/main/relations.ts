import { relations } from "drizzle-orm/relations";
import { usersInCredentials, history, songs, relationsProducer, singer, relationSinger, videoTypeLabelInInternal } from "./schema";

export const historyRelations = relations(history, ({one}) => ({
	usersInCredential: one(usersInCredentials, {
		fields: [history.changedBy],
		references: [usersInCredentials.unqId]
	}),
}));

export const usersInCredentialsRelations = relations(usersInCredentials, ({many}) => ({
	histories: many(history),
	videoTypeLabelInInternals: many(videoTypeLabelInInternal),
}));

export const relationsProducerRelations = relations(relationsProducer, ({one}) => ({
	song: one(songs, {
		fields: [relationsProducer.songId],
		references: [songs.id]
	}),
}));

export const songsRelations = relations(songs, ({many}) => ({
	relationsProducers: many(relationsProducer),
	relationSingers: many(relationSinger),
}));

export const relationSingerRelations = relations(relationSinger, ({one}) => ({
	singer: one(singer, {
		fields: [relationSinger.singerId],
		references: [singer.id]
	}),
	song: one(songs, {
		fields: [relationSinger.songId],
		references: [songs.id]
	}),
}));

export const singerRelations = relations(singer, ({many}) => ({
	relationSingers: many(relationSinger),
}));

export const videoTypeLabelInInternalRelations = relations(videoTypeLabelInInternal, ({one}) => ({
	usersInCredential: one(usersInCredentials, {
		fields: [videoTypeLabelInInternal.user],
		references: [usersInCredentials.unqId]
	}),
}));