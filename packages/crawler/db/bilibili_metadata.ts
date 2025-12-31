import {
	type BilibiliMetadataType,
	bilibiliMetadata,
	bilibiliUser,
	db,
	labellingResult,
} from "@core/drizzle";
import type { PartialBy } from "@core/lib";
import { eq, isNull } from "drizzle-orm";
import { AkariModelVersion } from "ml/const";

export async function insertIntoMetadata(
	data: PartialBy<BilibiliMetadataType, "id" | "createdAt" | "status">
) {
	await db.insert(bilibiliMetadata).values(data);
}

export async function videoExistsInAllData(aid: number) {
	const rows = await db
		.select({
			id: bilibiliMetadata.id,
		})
		.from(bilibiliMetadata)
		.where(eq(bilibiliMetadata.aid, aid))
		.limit(1);

	return rows.length > 0;
}

export async function userExistsInBiliUsers(uid: number) {
	const rows = await db
		.select({ id: bilibiliUser.id })
		.from(bilibiliUser)
		.where(eq(bilibiliUser.uid, uid))
		.limit(1);

	return rows.length > 0;
}

export async function getUnlabelledVideos() {
	const rows = await db
		.select({ aid: bilibiliMetadata.aid })
		.from(bilibiliMetadata)
		.leftJoin(labellingResult, eq(bilibiliMetadata.aid, labellingResult.aid))
		.where(isNull(labellingResult.aid));

	return rows.map((row) => row.aid);
}

export async function insertVideoLabel(aid: number, label: number) {
	await db
		.insert(labellingResult)
		.values({
			aid,
			label,
			modelVersion: AkariModelVersion,
		})
		.onConflictDoNothing({
			target: [labellingResult.aid, labellingResult.modelVersion],
		});
}

export async function getVideoInfoFromAllData(aid: number) {
	const rows = await db
		.select()
		.from(bilibiliMetadata)
		.where(eq(bilibiliMetadata.aid, aid))
		.limit(1);

	if (rows.length === 0) {
		return null;
	}

	const row = rows[0];
	return {
		description: row.description,
		tags: row.tags,
		title: row.title,
	};
}

export async function setBiliVideoStatus(aid: number, status: number) {
	await db.update(bilibiliMetadata).set({ status }).where(eq(bilibiliMetadata.aid, aid));
}

export async function getBiliVideoStatus(aid: number) {
	const rows = await db
		.select({ status: bilibiliMetadata.status })
		.from(bilibiliMetadata)
		.where(eq(bilibiliMetadata.aid, aid))
		.limit(1);

	if (rows.length === 0) return 0;
	return rows[0].status;
}
