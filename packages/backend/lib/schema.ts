import { z } from "zod";

const videoStatsSchema = z.object({
	aid: z.number(),
	coin: z.number(),
	danmaku: z.number(),
	favorite: z.number(),
	his_rank: z.number(),
	like: z.number(),
	now_rank: z.number(),
	reply: z.number(),
	share: z.number(),
	view: z.number(),
});

export const BiliAPIVideoMetadataSchema = z.object({
	aid: z.number(),
	bvid: z.string(),
	copyright: z.number(),
	ctime: z.number(),
	desc: z.string(),
	desc_v2: z.string(),
	duration: z.number(),
	owner: z.object({
		face: z.string(),
		mid: z.number(),
		name: z.string(),
	}),
	pic: z.string(),
	pubdate: z.number(),
	stat: videoStatsSchema,
	state: z.number(),
	tid: z.number(),
	tid_v2: z.number(),
	title: z.string(),
	tname: z.string(),
	tname_v2: z.string(),
});

export const BiliVideoSchema = z.object({
	aid: z.number(),
	bvid: z.string().nullable(),
	coverUrl: z.string().nullable(),
	createdAt: z.string().nullable(),
	description: z.string().nullable(),
	duration: z.number().nullable(),
	id: z.number(),
	publishedAt: z.string().nullable(),
	status: z.number(),
	tags: z.string().nullable(),
	title: z.string().nullable(),
	uid: z.number().nullable(),
});

export type BiliVideoType = z.infer<typeof BiliVideoSchema>;

export const SongSchema = z.object({
	aid: z.number().nullable(),
	createdAt: z.string(),
	deleted: z.boolean(),
	duration: z.number().nullable(),
	id: z.number(),
	image: z.string().nullable(),
	name: z.string().nullable(),
	neteaseId: z.number().nullable(),
	producer: z.string().nullable(),
	publishedAt: z.string().nullable(),
	type: z.number().nullable(),
	updatedAt: z.string(),
});
