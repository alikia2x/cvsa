import type { InferSelectModel } from "drizzle-orm";
import {
	usersInCredentials,
	bilibiliMetadata,
	latestVideoSnapshot,
	songs,
	videoSnapshot,
	producer,
	loginSessionsInCredentials
} from "./main/schema";

export type UserType = InferSelectModel<typeof usersInCredentials>;
export type SensitiveUserFields = "password" | "unqId";
export type BilibiliMetadataType = InferSelectModel<typeof bilibiliMetadata>;
export type VideoSnapshotType = InferSelectModel<typeof videoSnapshot>;
export type LatestVideoSnapshotType = InferSelectModel<typeof latestVideoSnapshot>;
export type SongType = InferSelectModel<typeof songs>;
export type ProducerType = InferSelectModel<typeof producer>;
export type SessionType = InferSelectModel<typeof loginSessionsInCredentials>;
