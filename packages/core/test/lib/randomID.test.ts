import { describe, expect, it } from "vitest";
import { generateRandomId } from "@core/lib/randomID.ts";

describe("generateRandomId", () => {
	it("should generate an ID of the specified length", () => {
		const length = 15;
		const id = generateRandomId(length);
		expect(id).toHaveLength(length);
	});

	it("should generate an ID containing only allowed characters", () => {
		const allowedChars = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
		const id = generateRandomId(20);
		for (const char of id) {
			expect(allowedChars).toContain(char);
		}
	});
});
