import { assertEquals } from "jsr:@std/assert";
import { videoTagsIsNull } from "lib/db/allData.ts";
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { postgresConfig } from "lib/db/pgConfig.ts";

// A minimal aid which has an empty tags field in our database
const TEST_AID = 63569;

Deno.test("videoTagsIsNull function", async () => {
	const client = new Client(postgresConfig);

	try {
		const transaction = client.createTransaction("test_transaction");
		await transaction.begin();

		const result1 = await videoTagsIsNull(transaction, TEST_AID);
		assertEquals(typeof result1, "boolean", "The result should be a boolean value.");
		assertEquals(result1, false, "The result should be false if tags is not NULL for the given aid.");

		await transaction.queryArray`UPDATE all_data SET tags = NULL WHERE aid = ${TEST_AID}`;

		const result2 = await videoTagsIsNull(transaction, TEST_AID);
		assertEquals(typeof result2, "boolean", "The result should be a boolean value.");
		assertEquals(result2, true, "The result should be true if tags is NULL for the given aid.");

		await transaction.rollback();
	} catch (error) {
		console.error("Error during test:", error);
		throw error;
	} finally {
		client.end();
	}
});
