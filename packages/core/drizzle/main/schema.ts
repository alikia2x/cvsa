import { pgTable, pgSchema, uniqueIndex, check, integer, text, timestamp, foreignKey, serial, bigint, jsonb, index, inet, varchar, smallint, real, boolean, bigserial, unique, vector, pgSequence } from "drizzle-orm/pg-core"
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
	check("users_id_not_null", sql`NOT NULL id`),
	check("users_username_not_null", sql`NOT NULL username`),
	check("users_password_not_null", sql`NOT NULL password`),
	check("users_unq_id_not_null", sql`NOT NULL unq_id`),
	check("users_role_not_null", sql`NOT NULL role`),
	check("users_created_at_not_null", sql`NOT NULL created_at`),
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
	check("history_id_not_null", sql`NOT NULL id`),
	check("history_object_id_not_null", sql`NOT NULL object_id`),
	check("history_change_type_not_null", sql`NOT NULL change_type`),
	check("history_changed_at_not_null", sql`NOT NULL changed_at`),
	check("history_changed_by_not_null", sql`NOT NULL changed_by`),
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
	check("login_sessions_id_not_null", sql`NOT NULL id`),
	check("login_sessions_uid_not_null", sql`NOT NULL uid`),
	check("login_sessions_created_at_not_null", sql`NOT NULL created_at`),
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
	check("bilibili_metadata_id_not_null", sql`NOT NULL id`),
	check("bilibili_metadata_aid_not_null", sql`NOT NULL aid`),
	check("bilibili_metadata_status_not_null", sql`NOT NULL status`),
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
	check("human_classified_lables_id_not_null", sql`NOT NULL id`),
	check("human_classified_lables_aid_not_null", sql`NOT NULL aid`),
	check("human_classified_lables_uid_not_null", sql`NOT NULL uid`),
	check("human_classified_lables_label_not_null", sql`NOT NULL label`),
	check("human_classified_lables_created_at_not_null", sql`NOT NULL created_at`),
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
	index("idx_vid_snapshot_aid_created_at").using("btree", table.aid.asc().nullsLast().op("int8_ops"), table.createdAt.asc().nullsLast().op("int8_ops")),
	index("idx_vid_snapshot_time").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	check("video_snapshot_id_not_null", sql`NOT NULL id`),
	check("video_snapshot_created_at_not_null", sql`NOT NULL created_at`),
	check("video_snapshot_views_not_null", sql`NOT NULL views`),
	check("video_snapshot_aid_not_null", sql`NOT NULL aid`),
]);

export const producer = pgTable("producer", {
	id: integer().primaryKey().notNull(),
	name: text().notNull(),
}, (table) => [
	check("producer_id_not_null", sql`NOT NULL id`),
	check("producer_name_not_null", sql`NOT NULL name`),
]);

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
	check("labelling_result_id_not_null", sql`NOT NULL id`),
	check("labelling_result_aid_not_null", sql`NOT NULL aid`),
	check("labelling_result_label_not_null", sql`NOT NULL label`),
	check("labelling_result_model_version_not_null", sql`NOT NULL model_version`),
	check("labelling_result_created_at_not_null", sql`NOT NULL created_at`),
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
	check("latest_video_snapshot_aid_not_null", sql`NOT NULL aid`),
	check("latest_video_snapshot_time_not_null", sql`NOT NULL "time"`),
	check("latest_video_snapshot_views_not_null", sql`NOT NULL views`),
	check("latest_video_snapshot_coins_not_null", sql`NOT NULL coins`),
	check("latest_video_snapshot_likes_not_null", sql`NOT NULL likes`),
	check("latest_video_snapshot_favorites_not_null", sql`NOT NULL favorites`),
	check("latest_video_snapshot_replies_not_null", sql`NOT NULL replies`),
	check("latest_video_snapshot_danmakus_not_null", sql`NOT NULL danmakus`),
	check("latest_video_snapshot_shares_not_null", sql`NOT NULL shares`),
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
	check("relations_producer_id_not_null", sql`NOT NULL id`),
	check("relations_producer_song_id_not_null", sql`NOT NULL song_id`),
	check("relations_producer_created_at_not_null", sql`NOT NULL created_at`),
	check("relations_producer_producer_id_not_null", sql`NOT NULL producer_id`),
	check("relations_producer_updated_at_not_null", sql`NOT NULL updated_at`),
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
	check("eta_aid_not_null", sql`NOT NULL aid`),
	check("eta_eta_not_null", sql`NOT NULL eta`),
	check("eta_speed_not_null", sql`NOT NULL speed`),
	check("eta_current_views_not_null", sql`NOT NULL current_views`),
	check("eta_updated_at_not_null", sql`NOT NULL updated_at`),
]);

export const singer = pgTable("singer", {
	id: serial().primaryKey().notNull(),
	name: text().notNull(),
}, (table) => [
	check("singer_id_not_null", sql`NOT NULL id`),
	check("singer_name_not_null", sql`NOT NULL name`),
]);

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
	check("songs_id_not_null", sql`NOT NULL id`),
	check("songs_created_at_not_null", sql`NOT NULL created_at`),
	check("songs_updated_at_not_null", sql`NOT NULL updated_at`),
	check("songs_deleted_not_null", sql`NOT NULL deleted`),
]);

export const snapshotSchedule = pgTable("snapshot_schedule", {
	id: bigserial({ mode: "bigint" }).notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	type: text(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	startedAt: timestamp("started_at", { withTimezone: true, mode: 'string' }),
	finishedAt: timestamp("finished_at", { withTimezone: true, mode: 'string' }),
	status: text().default('pending').notNull(),
}, (table) => [
	index("idx_snapshot_schedule_aid_status_type").using("btree", table.aid.asc().nullsLast().op("int8_ops"), table.status.asc().nullsLast().op("text_ops"), table.type.asc().nullsLast().op("text_ops")),
	index("idx_snapshot_schedule_status_started_at").using("btree", table.status.asc().nullsLast().op("timestamptz_ops"), table.startedAt.asc().nullsLast().op("text_ops")),
	uniqueIndex("snapshot_schedule_pkey").using("btree", table.id.asc().nullsLast().op("int8_ops")),
	check("snapshot_schedule_id_not_null", sql`NOT NULL id`),
	check("snapshot_schedule_aid_not_null", sql`NOT NULL aid`),
	check("snapshot_schedule_created_at_not_null", sql`NOT NULL created_at`),
	check("snapshot_schedule_status_not_null", sql`NOT NULL status`),
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
	check("relation_singer_id_not_null", sql`NOT NULL id`),
	check("relation_singer_song_id_not_null", sql`NOT NULL song_id`),
	check("relation_singer_singer_id_not_null", sql`NOT NULL singer_id`),
	check("relation_singer_created_at_not_null", sql`NOT NULL created_at`),
	check("relation_singer_updated_at_not_null", sql`NOT NULL updated_at`),
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
	check("bilibili_user_id_not_null", sql`NOT NULL id`),
	check("bilibili_user_uid_not_null", sql`NOT NULL uid`),
	check("bilibili_user_username_not_null", sql`NOT NULL username`),
	check("bilibili_user_desc_not_null", sql`NOT NULL "desc"`),
	check("bilibili_user_fans_not_null", sql`NOT NULL fans`),
	check("bilibili_user_created_at_not_null", sql`NOT NULL created_at`),
	check("bilibili_user_updated_at_not_null", sql`NOT NULL updated_at`),
]);

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
	check("video_type_label_id_not_null", sql`NOT NULL id`),
	check("video_type_label_aid_not_null", sql`NOT NULL aid`),
	check("video_type_label_label_not_null", sql`NOT NULL label`),
	check("video_type_label_user_not_null", sql`NOT NULL "user"`),
	check("video_type_label_created_at_not_null", sql`NOT NULL created_at`),
]);

export const embeddingsInInternal = internal.table("embeddings", {
	id: serial().primaryKey().notNull(),
	modelName: text("model_name").notNull(),
	dataChecksum: text("data_checksum").notNull(),
	vec2048: vector("vec_2048", { dimensions: 2048 }).notNull(),
	vec1536: vector("vec_1536", { dimensions: 1536 }).notNull(),
	vec1024: vector("vec_1024", { dimensions: 1024 }).notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`),
}, (table) => [
	unique("embeddings_data_checksum_key").on(table.dataChecksum),
	check("embeddings_id_not_null", sql`NOT NULL id`),
	check("embeddings_model_name_not_null", sql`NOT NULL model_name`),
	check("embeddings_data_checksum_not_null", sql`NOT NULL data_checksum`),
	check("embeddings_vec_2048_not_null", sql`NOT NULL vec_2048`),
	check("embeddings_vec_1536_not_null", sql`NOT NULL vec_1536`),
	check("embeddings_vec_1024_not_null", sql`NOT NULL vec_1024`),
]);
