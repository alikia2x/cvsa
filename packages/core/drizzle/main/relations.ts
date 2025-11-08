import { relations } from "drizzle-orm/relations";
import { songs, relationSinger, singer, relationsProducer } from "./schema";

export const relationSingerRelations = relations(relationSinger, ({one}) => ({
	song: one(songs, {
		fields: [relationSinger.songId],
		references: [songs.id]
	}),
	singer: one(singer, {
		fields: [relationSinger.singerId],
		references: [singer.id]
	}),
}));

export const songsRelations = relations(songs, ({many}) => ({
	relationSingers: many(relationSinger),
	relationsProducers: many(relationsProducer),
}));

export const singerRelations = relations(singer, ({many}) => ({
	relationSingers: many(relationSinger),
}));

export const relationsProducerRelations = relations(relationsProducer, ({one}) => ({
	song: one(songs, {
		fields: [relationsProducer.songId],
		references: [songs.id]
	}),
}));