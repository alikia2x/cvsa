import { z } from "zod";

const videoStatsSchema = z.object({
	aid: z.number(),
	view: z.number(),
	danmaku: z.number(),
	reply: z.number(),
	favorite: z.number(),
	coin: z.number(),
	share: z.number(),
	now_rank: z.number(),
	his_rank: z.number(),
	like: z.number(),
});

export const BiliAPIVideoMetadataSchema = z.object({
	bvid: z.string(),
	aid: z.number(),
	copyright: z.number(),
	pic: z.string(),
	title: z.string(),
	pubdate: z.number(),
	ctime: z.number(),
	desc: z.string(),
	desc_v2: z.string(),
	tname: z.string(),
	tid: z.number(),
	tid_v2: z.number(),
	tname_v2: z.string(),
	state: z.number(),
	duration: z.number(),
	owner: z.object({
		mid: z.number(),
		name: z.string(),
		face: z.string(),
	}),
	stat: videoStatsSchema,
});

export const BiliVideoSchema = z.object({
	duration: z.number().nullable(),
	id: z.number(),
	aid: z.number(),
	publishedAt: z.string().nullable(),
	createdAt: z.string().nullable(),
	description: z.string().nullable(),
	bvid: z.string().nullable(),
	uid: z.number().nullable(),
	tags: z.string().nullable(),
	title: z.string().nullable(),
	status: z.number(),
	coverUrl: z.string().nullable(),
});

export type BiliVideoType = z.infer<typeof BiliVideoSchema>;

export const SongSchema = z.object({
	duration: z.number().nullable(),
	name: z.string().nullable(),
	id: z.number(),
	aid: z.number().nullable(),
	publishedAt: z.string().nullable(),
	type: z.number().nullable(),
	neteaseId: z.number().nullable(),
	createdAt: z.string(),
	updatedAt: z.string(),
	deleted: z.boolean(),
	image: z.string().nullable(),
	producer: z.string().nullable(),
});
