import { db } from "@core/drizzle";
import { sql } from "drizzle-orm";

export const getClosestSnapshot = async (aid: number, targetTime: Date) => {
	const closest = await db.execute<{ created_at: Date; views: number }>(sql`
		SELECT created_at, views
		FROM (
			(SELECT created_at, views, 'later' AS type
			FROM video_snapshot
			WHERE aid = ${aid}
			AND created_at >= ${targetTime.toISOString()}
			ORDER BY created_at
			LIMIT 1)
			UNION ALL
			(SELECT created_at, views, 'earlier' AS type
			FROM video_snapshot
			WHERE aid = ${aid}
			AND created_at <= ${targetTime.toISOString()}
			ORDER BY created_at DESC
			LIMIT 1)
		) AS combined
		ORDER BY 
			CASE 
				WHEN created_at >= ${targetTime.toISOString()} THEN created_at -${targetTime.toISOString()}
				ELSE ${targetTime.toISOString()} - created_at
			END
		LIMIT 1;
	`);
	return closest[0] || null;
};
