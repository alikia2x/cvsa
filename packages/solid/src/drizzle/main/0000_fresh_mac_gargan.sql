-- Current sql file was generated after introspecting the database
-- If you want to run this migration please uncomment this code before executing migrations
/*
CREATE SEQUENCE "public"."all_data_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE SEQUENCE "public"."labeling_result_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE SEQUENCE "public"."songs_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE SEQUENCE "public"."video_snapshot_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE SEQUENCE "public"."views_increment_rate_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START WITH 1 CACHE 1;--> statement-breakpoint
CREATE TABLE "content" (
	"page_id" text PRIMARY KEY NOT NULL,
	"page_content" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp with time zone,
	"deleted_at" timestamp with time zone
);
--> statement-breakpoint
CREATE TABLE "bilibili_user" (
	"id" serial PRIMARY KEY NOT NULL,
	"uid" bigint NOT NULL,
	"username" text NOT NULL,
	"desc" text NOT NULL,
	"fans" integer NOT NULL,
	CONSTRAINT "unq_bili-user_uid" UNIQUE("uid")
);
--> statement-breakpoint
CREATE TABLE "bilibili_metadata" (
	"id" integer DEFAULT nextval('all_data_id_seq'::regclass) NOT NULL,
	"aid" bigint NOT NULL,
	"bvid" varchar(12),
	"description" text,
	"uid" bigint,
	"tags" text,
	"title" text,
	"published_at" timestamp with time zone,
	"duration" integer,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
	"status" integer DEFAULT 0 NOT NULL,
	"cover_url" text
);
--> statement-breakpoint
CREATE TABLE "classified_labels_human" (
	"id" serial PRIMARY KEY NOT NULL,
	"aid" bigint NOT NULL,
	"author" uuid NOT NULL,
	"label" smallint NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE "labelling_result" (
	"id" integer DEFAULT nextval('labeling_result_id_seq'::regclass) NOT NULL,
	"aid" bigint NOT NULL,
	"label" smallint NOT NULL,
	"model_version" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"logits" smallint[]
);
--> statement-breakpoint
CREATE TABLE "latest_video_snapshot" (
	"aid" bigint PRIMARY KEY NOT NULL,
	"time" timestamp with time zone NOT NULL,
	"views" integer NOT NULL,
	"coins" integer NOT NULL,
	"likes" integer NOT NULL,
	"favorites" integer NOT NULL,
	"replies" integer NOT NULL,
	"danmakus" integer NOT NULL,
	"shares" integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE "video_snapshot" (
	"id" integer DEFAULT nextval('video_snapshot_id_seq'::regclass) NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"views" integer NOT NULL,
	"coins" integer NOT NULL,
	"likes" integer NOT NULL,
	"favorites" integer NOT NULL,
	"shares" integer NOT NULL,
	"danmakus" integer NOT NULL,
	"aid" bigint NOT NULL,
	"replies" integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE "snapshot_schedule" (
	"id" bigserial NOT NULL,
	"aid" bigint NOT NULL,
	"type" text,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"started_at" timestamp with time zone,
	"finished_at" timestamp with time zone,
	"status" text DEFAULT 'pending' NOT NULL
);
--> statement-breakpoint
CREATE TABLE "songs" (
	"id" integer DEFAULT nextval('songs_id_seq'::regclass) NOT NULL,
	"name" text,
	"aid" bigint,
	"published_at" timestamp with time zone,
	"duration" integer,
	"type" smallint,
	"romanized_name" text,
	"netease_id" bigint,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"deleted" boolean DEFAULT false NOT NULL
);
--> statement-breakpoint
CREATE TABLE "views_increment_rate" (
	"id" integer DEFAULT nextval('views_increment_rate_id_seq'::regclass) NOT NULL,
	"aid" bigint NOT NULL,
	"old_time" timestamp with time zone NOT NULL,
	"new_time" timestamp with time zone NOT NULL,
	"old_views" integer NOT NULL,
	"new_views" integer NOT NULL,
	"interval" interval NOT NULL,
	"updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
	"speed" real
);
--> statement-breakpoint
CREATE INDEX "idx_content_created-at" ON "content" USING btree ("created_at" timestamptz_ops);--> statement-breakpoint
CREATE INDEX "idx_bili-user_uid" ON "bilibili_user" USING btree ("uid" int8_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "all_data_pkey" ON "bilibili_metadata" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE INDEX "idx_all-data_aid" ON "bilibili_metadata" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_all-data_bvid" ON "bilibili_metadata" USING btree ("bvid" text_ops);--> statement-breakpoint
CREATE INDEX "idx_all-data_uid" ON "bilibili_metadata" USING btree ("uid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_bili-meta_status" ON "bilibili_metadata" USING btree ("status" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "unq_all-data_aid" ON "bilibili_metadata" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_classified-labels-human_aid" ON "classified_labels_human" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_classified-labels-human_author" ON "classified_labels_human" USING btree ("author" uuid_ops);--> statement-breakpoint
CREATE INDEX "idx_classified-labels-human_created-at" ON "classified_labels_human" USING btree ("created_at" timestamptz_ops);--> statement-breakpoint
CREATE INDEX "idx_classified-labels-human_label" ON "classified_labels_human" USING btree ("label" int2_ops);--> statement-breakpoint
CREATE INDEX "idx_labeling_label_model-version" ON "labelling_result" USING btree ("label" int2_ops,"model_version" int2_ops);--> statement-breakpoint
CREATE INDEX "idx_labeling_model-version" ON "labelling_result" USING btree ("model_version" text_ops);--> statement-breakpoint
CREATE INDEX "idx_labelling_aid-label" ON "labelling_result" USING btree ("aid" int2_ops,"label" int2_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "labeling_result_pkey" ON "labelling_result" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "unq_labelling-result_aid_model-version" ON "labelling_result" USING btree ("aid" int8_ops,"model_version" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_latest-video-snapshot_time" ON "latest_video_snapshot" USING btree ("time" timestamptz_ops);--> statement-breakpoint
CREATE INDEX "idx_latest-video-snapshot_views" ON "latest_video_snapshot" USING btree ("views" int4_ops);--> statement-breakpoint
CREATE INDEX "idx_vid_snapshot_aid" ON "video_snapshot" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_vid_snapshot_time" ON "video_snapshot" USING btree ("created_at" timestamptz_ops);--> statement-breakpoint
CREATE INDEX "idx_vid_snapshot_views" ON "video_snapshot" USING btree ("views" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "video_snapshot_pkey" ON "video_snapshot" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE INDEX "idx_snapshot_schedule_aid" ON "snapshot_schedule" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_snapshot_schedule_started_at" ON "snapshot_schedule" USING btree ("started_at" timestamptz_ops);--> statement-breakpoint
CREATE INDEX "idx_snapshot_schedule_status" ON "snapshot_schedule" USING btree ("status" text_ops);--> statement-breakpoint
CREATE INDEX "idx_snapshot_schedule_type" ON "snapshot_schedule" USING btree ("type" text_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "snapshot_schedule_pkey" ON "snapshot_schedule" USING btree ("id" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_aid" ON "songs" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_hash_songs_aid" ON "songs" USING hash ("aid" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_netease_id" ON "songs" USING btree ("netease_id" int8_ops);--> statement-breakpoint
CREATE INDEX "idx_published_at" ON "songs" USING btree ("published_at" timestamptz_ops);--> statement-breakpoint
CREATE INDEX "idx_type" ON "songs" USING btree ("type" int2_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "songs_pkey" ON "songs" USING btree ("id" int4_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "unq_songs_aid" ON "songs" USING btree ("aid" int8_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "unq_songs_netease_id" ON "songs" USING btree ("netease_id" int8_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "unq_views-increment-rate_aid_interval" ON "views_increment_rate" USING btree ("aid" int8_ops,"interval" int8_ops);--> statement-breakpoint
CREATE UNIQUE INDEX "views_increment_rate_pkey" ON "views_increment_rate" USING btree ("id" int4_ops);
*/