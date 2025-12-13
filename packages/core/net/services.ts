import networkDelegate, { type RequestTasks } from "@core/net/delegate.ts";
import { VideoInfoResponse } from "@core/net/bilibili";
import { PartialBy } from "@core/lib";
import { VideoSnapshotType } from "@core/drizzle";

export class BilibiliService {
	private static videoMetadataUrl = "https://api.bilibili.com/x/web-interface/view";

	private static async getVideoMetadata(aid: number, task: RequestTasks) {
		const url = new URL(this.videoMetadataUrl);
		url.searchParams.set("aid", aid.toString());
		return networkDelegate.request<VideoInfoResponse>(url.toString(), task);
	}

	static async milestoneSnapshot(aid: number): Promise<PartialBy<VideoSnapshotType, "id">> {
		const metadata = await this.getVideoMetadata(aid, "snapshotMilestoneVideo");
		const stats = metadata.data.data.stat;
		return {
			aid,
			createdAt: new Date(metadata.time).toISOString(),
			views: stats.view,
			likes: stats.like,
			coins: stats.coin,
			favorites: stats.favorite,
			replies: stats.reply,
			shares: stats.share,
			danmakus: stats.danmaku
		};
	}
}
