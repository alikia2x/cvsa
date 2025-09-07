import type { InferSelectModel } from "drizzle-orm";
import { users } from "~db/cred/schema";
import { bilibiliMetadata, latestVideoSnapshot, songs, videoSnapshot } from "~db/main/schema";

export type UserType = InferSelectModel<typeof users>;
export type SensitiveUserFields = "password" | "unqId";
export type BilibiliMetadataType = InferSelectModel<typeof bilibiliMetadata>;
export type VideoSnapshotType = InferSelectModel<typeof videoSnapshot>;
export type LatestVideoSnapshotType = InferSelectModel<typeof latestVideoSnapshot>;
export type SongType = InferSelectModel<typeof songs>;