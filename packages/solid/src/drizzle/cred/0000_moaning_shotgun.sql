-- Current sql file was generated after introspecting the database
-- If you want to run this migration please uncomment this code before executing migrations
/*
CREATE SEQUENCE "public"."captcha_difficulty_settings_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE SEQUENCE "public"."users_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE TABLE "captcha_difficulty_settings" (
	"id" integer DEFAULT nextval('captcha_difficulty_settings_id_seq'::regclass) NOT NULL,
	"method" text NOT NULL,
	"path" text NOT NULL,
	"duration" real NOT NULL,
	"threshold" integer NOT NULL,
	"difficulty" integer NOT NULL,
	"global" boolean NOT NULL
);
--> statement-breakpoint
CREATE TABLE "login_sessions" (
	"id" text NOT NULL,
	"uid" integer NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"expire_at" timestamp with time zone,
	"last_used_at" timestamp with time zone,
	"ip_address" "inet",
	"user_agent" text,
	"deactivated_at" timestamp with time zone
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" integer DEFAULT nextval('users_id_seq'::regclass) NOT NULL,
	"nickname" text,
	"username" text NOT NULL,
	"password" text NOT NULL,
	"unq_id" text DEFAULT gen_random_uuid() NOT NULL,
	"role" text DEFAULT 'USER' NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX "captcha_difficulty_settings_pkey" ON "captcha_difficulty_settings" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE INDEX "inx_login-sessions_uid" ON "login_sessions" USING btree ("uid" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "login_sessions_pkey" ON "login_sessions" USING btree ("id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "users_pkey" ON "users" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "users_pkey1" ON "users" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "users_unq_id_key" ON "users" USING btree ("unq_id" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "users_username_key" ON "users" USING btree ("username" text_ops);
*/