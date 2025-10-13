import { Psql } from "@core/db/psql";

export async function updateETA(sql: Psql, aid: number, eta: number, speed: number, views: number) {
	return sql`
		INSERT INTO eta (aid, eta, speed, current_views)
		VALUES (${aid}, ${eta}, ${speed}, ${views})
		ON CONFLICT (aid)
		DO UPDATE SET eta = ${eta}, speed = ${speed}, current_views = ${views}, updated_at = now()
	`;
}
