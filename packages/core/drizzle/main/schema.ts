import { pgTable, pgSchema, uniqueIndex, integer, text, timestamp, foreignKey, serial, bigint, jsonb, index, inet, varchar, smallint, real, boolean, unique, doublePrecision, vector, bigserial, pgView, numeric, pgSequence } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"

export const credentials = pgSchema("credentials");
export const internal = pgSchema("internal");
export const userRoleInCredentials = credentials.enum("user_role", ['ADMIN', 'USER', 'OWNER'])

export const usersIdSeqInCredentials = credentials.sequence("users_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const allDataIdSeq = pgSequence("all_data_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const labelingResultIdSeq = pgSequence("labeling_result_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const relationSingerIdSeq = pgSequence("relation_singer_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const relationsProducerIdSeq = pgSequence("relations_producer_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const songsIdSeq = pgSequence("songs_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const videoSnapshotIdSeq = pgSequence("video_snapshot_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const viewsIncrementRateIdSeq = pgSequence("views_increment_rate_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "9223372036854775807", cache: "1", cycle: false })

export const usersInCredentials = credentials.table("users", {
	id: integer().default(sql`nextval('credentials.users_id_seq'::regclass)`).notNull(),
	nickname: text(),
	username: text().notNull(),
	password: text().notNull(),
	unqId: text("unq_id").notNull(),
	role: userRoleInCredentials().default('USER').notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	uniqueIndex("users_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	uniqueIndex("users_pkey1").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	uniqueIndex("users_username_key").using("btree", table.username.asc().nullsLast().op("text_ops")),
]);

export const history = pgTable("history", {
	id: serial().primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	objectId: bigint("object_id", { mode: "number" }).notNull(),
	changeType: text("change_type").notNull(),
	changedAt: timestamp("changed_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	changedBy: text("changed_by").notNull(),
	data: jsonb(),
}, (table) => [
	foreignKey({
			columns: [table.changedBy],
			foreignColumns: [usersInCredentials.unqId],
			name: "rel_history_changed_by"
		}),
]);

export const loginSessionsInCredentials = credentials.table("login_sessions", {
	id: text().notNull(),
	uid: integer().notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	expireAt: timestamp("expire_at", { withTimezone: true, mode: 'string' }),
	lastUsedAt: timestamp("last_used_at", { withTimezone: true, mode: 'string' }),
	ipAddress: inet("ip_address"),
	userAgent: text("user_agent"),
	deactivatedAt: timestamp("deactivated_at", { withTimezone: true, mode: 'string' }),
}, (table) => [
	index("inx_login-sessions_uid").using("btree", table.uid.asc().nullsLast().op("int4_ops")),
	uniqueIndex("login_sessions_pkey").using("btree", table.id.asc().nullsLast().op("text_ops")),
]);

export const videoSnapshot = pgTable("video_snapshot", {
	id: integer().default(sql`nextval('video_snapshot_id_seq'::regclass)`).notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	views: integer().notNull(),
	coins: integer(),
	likes: integer(),
	favorites: integer(),
	shares: integer(),
	danmakus: integer(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	replies: integer(),
}, (table) => [
	index("video_snapshot_new_aid_created_at_idx").using("btree", table.aid.asc().nullsLast().op("int8_ops"), table.createdAt.asc().nullsLast().op("int8_ops")),
	index("video_snapshot_new_created_at_idx").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
]);

export const bilibiliMetadata = pgTable("bilibili_metadata", {
	id: integer().default(sql`nextval('all_data_id_seq'::regclass)`).notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	bvid: varchar({ length: 12 }),
	description: text(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	uid: bigint({ mode: "number" }),
	tags: text(),
	title: text(),
	publishedAt: timestamp("published_at", { withTimezone: true, mode: 'string' }),
	duration: integer(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	status: integer().default(0).notNull(),
	coverUrl: text("cover_url"),
}, (table) => [
	uniqueIndex("all_data_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	index("idx_all-data_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_all-data_bvid").using("btree", table.bvid.asc().nullsLast().op("text_ops")),
	index("idx_all-data_uid").using("btree", table.uid.asc().nullsLast().op("int8_ops")),
	index("idx_bili-meta_status").using("btree", table.status.asc().nullsLast().op("int4_ops")),
	uniqueIndex("unq_all-data_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
]);

export const humanClassifiedLables = pgTable("human_classified_lables", {
	id: serial().primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	uid: integer().notNull(),
	label: smallint().notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	index("idx_classified-labels-human_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_classified-labels-human_author").using("btree", table.uid.asc().nullsLast().op("int4_ops")),
	index("idx_classified-labels-human_created-at").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_classified-labels-human_label").using("btree", table.label.asc().nullsLast().op("int2_ops")),
]);

export const videoSnapshotBackup = pgTable("video_snapshot_backup", {
	id: integer().default(sql`nextval('video_snapshot_id_seq'::regclass)`).notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	views: integer().notNull(),
	coins: integer(),
	likes: integer(),
	favorites: integer(),
	shares: integer(),
	danmakus: integer(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	replies: integer(),
}, (table) => [
	index("idx_vid_snapshot_aid_created_at").using("btree", table.aid.asc().nullsLast().op("int8_ops"), table.createdAt.asc().nullsLast().op("int8_ops")),
	index("idx_vid_snapshot_time").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
]);

export const producer = pgTable("producer", {
	id: integer().primaryKey().notNull(),
	name: text().notNull(),
});

export const labellingResult = pgTable("labelling_result", {
	id: integer().default(sql`nextval('labeling_result_id_seq'::regclass)`).notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	label: smallint().notNull(),
	modelVersion: text("model_version").notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	logits: smallint().array(),
}, (table) => [
	index("idx_labeling_label_model-version").using("btree", table.label.asc().nullsLast().op("int2_ops"), table.modelVersion.asc().nullsLast().op("int2_ops")),
	index("idx_labeling_model-version").using("btree", table.modelVersion.asc().nullsLast().op("text_ops")),
	index("idx_labelling_aid-label").using("btree", table.aid.asc().nullsLast().op("int2_ops"), table.label.asc().nullsLast().op("int2_ops")),
	uniqueIndex("labeling_result_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	uniqueIndex("unq_labelling-result_aid_model-version").using("btree", table.aid.asc().nullsLast().op("int8_ops"), table.modelVersion.asc().nullsLast().op("int8_ops")),
]);

export const latestVideoSnapshot = pgTable("latest_video_snapshot", {
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).primaryKey().notNull(),
	time: timestamp({ withTimezone: true, mode: 'string' }).notNull(),
	views: integer().notNull(),
	coins: integer().notNull(),
	likes: integer().notNull(),
	favorites: integer().notNull(),
	replies: integer().notNull(),
	danmakus: integer().notNull(),
	shares: integer().notNull(),
}, (table) => [
	index("idx_latest-video-snapshot_time").using("btree", table.time.asc().nullsLast().op("timestamptz_ops")),
	index("idx_latest-video-snapshot_views").using("btree", table.views.asc().nullsLast().op("int4_ops")),
]);

export const relationsProducer = pgTable("relations_producer", {
	id: integer().default(sql`nextval('relations_producer_id_seq'::regclass)`).primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	songId: bigint("song_id", { mode: "number" }).notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	producerId: integer("producer_id").notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	foreignKey({
			columns: [table.songId],
			foreignColumns: [songs.id],
			name: "fkey_relations_producer_songs_id"
		}),
]);

export const eta = pgTable("eta", {
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).primaryKey().notNull(),
	eta: real().notNull(),
	speed: real().notNull(),
	currentViews: integer("current_views").notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).defaultNow().notNull(),
}, (table) => [
	index("idx_eta_eta_current_views").using("btree", table.eta.asc().nullsLast().op("int4_ops"), table.currentViews.asc().nullsLast().op("int4_ops")),
]);

export const singer = pgTable("singer", {
	id: serial().primaryKey().notNull(),
	name: text().notNull(),
});

export const songs = pgTable("songs", {
	id: integer().default(sql`nextval('songs_id_seq'::regclass)`).notNull(),
	name: text(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }),
	publishedAt: timestamp("published_at", { withTimezone: true, mode: 'string' }),
	duration: integer(),
	type: smallint(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	neteaseId: bigint("netease_id", { mode: "number" }),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	deleted: boolean().default(false).notNull(),
	image: text(),
	producer: text(),
}, (table) => [
	index("idx_hash_songs_aid").using("hash", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_netease_id").using("btree", table.neteaseId.asc().nullsLast().op("int8_ops")),
	index("idx_published_at").using("btree", table.publishedAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_songs_name").using("gin", table.name.asc().nullsLast().op("gin_trgm_ops")),
	index("idx_type").using("btree", table.type.asc().nullsLast().op("int2_ops")),
	uniqueIndex("unq_songs_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	uniqueIndex("unq_songs_netease_id").using("btree", table.neteaseId.asc().nullsLast().op("int8_ops")),
]);

export const relationSinger = pgTable("relation_singer", {
	id: integer().default(sql`nextval('relation_singer_id_seq'::regclass)`).primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	songId: bigint("song_id", { mode: "number" }).notNull(),
	singerId: integer("singer_id").notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	foreignKey({
			columns: [table.singerId],
			foreignColumns: [singer.id],
			name: "fkey_singer_id"
		}),
	foreignKey({
			columns: [table.songId],
			foreignColumns: [songs.id],
			name: "fkey_song_id"
		}),
]);

export const bilibiliUser = pgTable("bilibili_user", {
	id: serial().primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	uid: bigint({ mode: "number" }).notNull(),
	username: text().notNull(),
	desc: text().notNull(),
	fans: integer().notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	avatar: text(),
}, (table) => [
	index("idx_bili-user_uid").using("btree", table.uid.asc().nullsLast().op("int8_ops")),
	unique("unq_bili-user_uid").on(table.uid),
]);

export const etaInInternal = internal.table("eta", {
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).primaryKey().notNull(),
	increment15M: doublePrecision("increment_15m"),
	increment2H: doublePrecision("increment_2h"),
	increment5H: doublePrecision("increment_5h"),
	increment12H: doublePrecision("increment_12h"),
	increment1D: doublePrecision("increment_1d"),
	increment3D: doublePrecision("increment_3d"),
	increment7D: doublePrecision("increment_7d"),
	increment14D: doublePrecision("increment_14d"),
	increment30D: doublePrecision("increment_30d"),
	targetMilestone: integer("target_milestone"),
	etaHours: doublePrecision("eta_hours"),
	nextSnapshot: timestamp("next_snapshot", { withTimezone: true, mode: 'string' }),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }),
});

export const videoTypeLabelInInternal = internal.table("video_type_label", {
	id: serial().primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	label: boolean().notNull(),
	user: text().notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	foreignKey({
			columns: [table.user],
			foreignColumns: [usersInCredentials.unqId],
			name: "fkey_video_type_label_user"
		}),
]);

export const embeddingsInInternal = internal.table("embeddings", {
	id: serial().primaryKey().notNull(),
	modelName: text("model_name").notNull(),
	dataChecksum: text("data_checksum").notNull(),
	vec2048: vector("vec_2048", { dimensions: 2048 }),
	vec1536: vector("vec_1536", { dimensions: 1536 }),
	vec1024: vector("vec_1024", { dimensions: 1024 }),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
	dimensions: smallint().notNull(),
}, (table) => [
	unique("embeddings_unique_model_dimensions_checksum").on(table.modelName, table.dataChecksum, table.dimensions),
]);

export const snapshotSchedule = pgTable("snapshot_schedule", {
	id: bigserial({ mode: "number" }).notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	type: text(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	startedAt: timestamp("started_at", { withTimezone: true, mode: 'string' }),
	finishedAt: timestamp("finished_at", { withTimezone: true, mode: 'string' }),
	status: text().default('pending').notNull(),
	startedAt5MinUtc: timestamp("started_at_5min_utc", { mode: 'string' }).generatedAlwaysAs(sql`(date_trunc('hour'::text, (started_at AT TIME ZONE 'UTC'::text)) + ((((EXTRACT(minute FROM (started_at AT TIME ZONE 'UTC'::text)))::integer / 5))::double precision * '00:05:00'::interval))`),
}, (table) => [
	index("idx_snapshot_schedule_aid_status_type").using("btree", table.aid.asc().nullsLast().op("int8_ops"), table.status.asc().nullsLast().op("text_ops"), table.type.asc().nullsLast().op("int8_ops")),
	index("idx_snapshot_schedule_pending_5min").using("btree", table.status.asc().nullsLast().op("text_ops"), table.startedAt5MinUtc.asc().nullsLast().op("timestamp_ops")).where(sql`(status = 'pending'::text)`),
	index("idx_snapshot_schedule_status_started_at").using("btree", table.status.asc().nullsLast().op("timestamptz_ops"), table.startedAt.asc().nullsLast().op("text_ops")),
	uniqueIndex("snapshot_schedule_pkey").using("btree", table.id.asc().nullsLast().op("int8_ops")),
]);