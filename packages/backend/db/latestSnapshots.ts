import { sql } from "@core/db/dbNew";
import type { LatestSnapshotType } from "@core/db/schema";

export async function getVideosInViewsRange(minViews: number, maxViews: number) {
	return sql<LatestSnapshotType[]>`
        SELECT *
        FROM latest_video_snapshot
        WHERE views >= ${minViews}
        AND views <= ${maxViews}
        ORDER BY views DESC
        LIMIT 5000
	`;
}
