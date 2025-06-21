import { pgTable, uniqueIndex, integer, text, real, boolean, index, timestamp, inet, pgSequence } from "drizzle-orm/pg-core"
import { sql } from "drizzle-orm"


export const captchaDifficultySettingsIdSeq = pgSequence("captcha_difficulty_settings_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })
export const usersIdSeq = pgSequence("users_id_seq", {  startWith: "1", increment: "1", minValue: "1", maxValue: "2147483647", cache: "1", cycle: false })

export const captchaDifficultySettings = pgTable("captcha_difficulty_settings", {
	id: integer().default(sql`nextval('captcha_difficulty_settings_id_seq'::regclass)`).notNull(),
	method: text().notNull(),
	path: text().notNull(),
	duration: real().notNull(),
	threshold: integer().notNull(),
	difficulty: integer().notNull(),
	global: boolean().notNull(),
}, (table) => [
	uniqueIndex("captcha_difficulty_settings_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
]);

export const loginSessions = pgTable("login_sessions", {
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

export const users = pgTable("users", {
	id: integer().default(sql`nextval('users_id_seq'::regclass)`).notNull(),
	nickname: text(),
	username: text().notNull(),
	password: text().notNull(),
	unqId: text("unq_id").default(sql`gen_random_uuid()`).notNull(),
	role: text().default('USER').notNull(),
	createdAt: timestamp("created_at", { withTimezone: true, mode: 'string' }).default(sql`CURRENT_TIMESTAMP`).notNull(),
}, (table) => [
	uniqueIndex("users_pkey").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	uniqueIndex("users_pkey1").using("btree", table.id.asc().nullsLast().op("int4_ops")),
	uniqueIndex("users_unq_id_key").using("btree", table.unqId.asc().nullsLast().op("text_ops")),
	uniqueIndex("users_username_key").using("btree", table.username.asc().nullsLast().op("text_ops")),
]);
