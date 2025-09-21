import { pgTable, uniqueIndex, index, integer, bigint, varchar, text, timestamp, smallint, boolean, unique, serial, bigserial, uuid, pgSequence } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"


export const allDataIdSeq = pgSequence("all_data_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const labelingResultIdSeq = pgSequence("labeling_result_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const songsIdSeq = pgSequence("songs_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const videoSnapshotIdSeq = pgSequence("video_snapshot_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const viewsIncrementRateIdSeq = pgSequence("views_increment_rate_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "9223372036854775807", cache: "1", cycle: false })

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

export const videoSnapshot = pgTable("video_snapshot", {
	id: integer().default(sql`nextval('video_snapshot_id_seq'::regclass)`).notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	views: integer().notNull(),
	coins: integer().notNull(),
	likes: integer().notNull(),
	favorites: integer().notNull(),
	shares: integer().notNull(),
	danmakus: integer().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	replies: integer().notNull(),
}, (table) => [
	index("idx_vid_snapshot_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_vid_snapshot_aid_created_at").using("btree", table.aid.asc().nullsLast().op("timestamptz_ops"), table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_vid_snapshot_time").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_vid_snapshot_views").using("btree", table.views.asc().nullsLast().op("int4_ops")),
	uniqueIndex("video_snapshot_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
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
	index("idx_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_hash_songs_aid").using("hash", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_netease_id").using("btree", table.neteaseId.asc().nullsLast().op("int8_ops")),
	index("idx_published_at").using("btree", table.publishedAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_type").using("btree", table.type.asc().nullsLast().op("int2_ops")),
	uniqueIndex("songs_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	uniqueIndex("unq_songs_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	uniqueIndex("unq_songs_netease_id").using("btree", table.neteaseId.asc().nullsLast().op("int8_ops")),
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
}, (table) => [
	index("idx_bili-user_uid").using("btree", table.uid.asc().nullsLast().op("int8_ops")),
	unique("unq_bili-user_uid").on(table.uid),
]);

export const singer = pgTable("singer", {
	id: serial().primaryKey().notNull(),
	name: text().notNull(),
});

export const relations = pgTable("relations", {
	id: serial().primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	sourceId: bigint("source_id", { mode: "number" }).notNull(),
	sourceType: text("source_type").notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	targetId: bigint("target_id", { mode: "number" }).notNull(),
	targetType: text("target_type").notNull(),
	relation: text().notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
	updatedAt: timestamp("updated_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	index("idx_relations_source_id_source_type_relation").using("btree", table.sourceId.asc().nullsLast().op("int8_ops"), table.sourceType.asc().nullsLast().op("int8_ops"), table.relation.asc().nullsLast().op("text_ops")),
	index("idx_relations_target_id_target_type_relation").using("btree", table.targetId.asc().nullsLast().op("text_ops"), table.targetType.asc().nullsLast().op("text_ops"), table.relation.asc().nullsLast().op("text_ops")),
	unique("unq_relations").on(table.sourceId, table.sourceType, table.targetId, table.targetType, table.relation),
]);

export const globalKv = pgTable("global_kv", {
	key: text().primaryKey().notNull(),
	value: text().notNull(),
});

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
	index("idx_snapshot_schedule_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_snapshot_schedule_started_at").using("btree", table.startedAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_snapshot_schedule_status").using("btree", table.status.asc().nullsLast().op("text_ops")),
	index("idx_snapshot_schedule_type").using("btree", table.type.asc().nullsLast().op("text_ops")),
	uniqueIndex("snapshot_schedule_pkey").using("btree", table.id.asc().nullsLast().op("int8_ops")),
]);

export const classifiedLabelsHuman = pgTable("classified_labels_human", {
	id: serial().primaryKey().notNull(),
	// You can use { mode: "bigint" } if numbers are exceeding js number limitations
	aid: bigint({ mode: "number" }).notNull(),
	author: uuid().notNull(),
	label: smallint().notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	index("idx_classified-labels-human_aid").using("btree", table.aid.asc().nullsLast().op("int8_ops")),
	index("idx_classified-labels-human_author").using("btree", table.author.asc().nullsLast().op("uuid_ops")),
	index("idx_classified-labels-human_created-at").using("btree", table.createdAt.asc().nullsLast().op("timestamptz_ops")),
	index("idx_classified-labels-human_label").using("btree", table.label.asc().nullsLast().op("int2_ops")),
]);
