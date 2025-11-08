import { z } from "zod";

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
	coverUrl: z.string().nullable()
});

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
	producer: z.string().nullable()
});
