import { assertEquals, assertInstanceOf, assertNotEquals } from "@std/assert";
import { findClosestSnapshot } from "lib/db/snapshotSchedule.ts";
import { postgresConfig } from "lib/db/pgConfig.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";

Deno.test("Snapshot Schedule - getShortTermTimeFeaturesForVideo", async () => {
	const client = new Client(postgresConfig);
	try {
		const result = await findClosestSnapshot(client, 247308539, new Date(1741983383000));
		assertNotEquals(result, null);
		const created_at = result!.created_at;
		const views = result!.views;
		assertInstanceOf(created_at, Date);
		assertEquals(typeof views, "number");
	} finally {
		client.end();
	}
});
