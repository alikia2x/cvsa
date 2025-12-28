import { describe, expect, test } from "bun:test";
import { getVideoInfo } from "@core/net/getVideoInfo";
import { bulkGetVideoStats } from "net/bulkGetVideoStats";

describe("Bilibili API", () => {
	test("bulkGetVideoStats()", async () => {
		const res = await bulkGetVideoStats([2]);
		expect(res).toBeObject();
	});
	test("bulkGetVideoStats()", async () => {
		const res = await getVideoInfo(2, "snapshotMilestoneVideo");
		expect(res).toBeObject();
	});
});
