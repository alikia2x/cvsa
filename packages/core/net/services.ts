import type { VideoSnapshotType } from "@core/drizzle";
import type { PartialBy } from "@core/lib";
import type { VideoInfoResponse } from "@core/net/bilibili";
import networkDelegate, { type RequestTasks } from "@core/net/delegate.ts";

export class BilibiliService {
	private static videoMetadataUrl = "https://api.bilibili.com/x/web-interface/view";

	private static async getVideoMetadata(aid: number, task: RequestTasks) {
		const url = new URL(BilibiliService.videoMetadataUrl);
		url.searchParams.set("aid", aid.toString());
		return networkDelegate.request<VideoInfoResponse>(url.toString(), task);
	}

	static async milestoneSnapshot(aid: number): Promise<PartialBy<VideoSnapshotType, "id">> {
		const metadata = await BilibiliService.getVideoMetadata(aid, "snapshotMilestoneVideo");
		const stats = metadata.data.data.stat;
		return {
			aid,
			coins: stats.coin,
			createdAt: new Date(metadata.time).toISOString(),
			danmakus: stats.danmaku,
			favorites: stats.favorite,
			likes: stats.like,
			replies: stats.reply,
			shares: stats.share,
			views: stats.view,
		};
	}
}
