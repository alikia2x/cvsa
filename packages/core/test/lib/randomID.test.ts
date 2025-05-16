import { describe, expect, it } from "vitest";
import { generateRandomId, decodeTimestampFromId } from "@core/lib/randomID.ts";

describe("generateRandomId", () => {
	it("should throw an error if the requested length is less than 8", () => {
		expect(() => generateRandomId(7)).toThrowError("Length must be at least 8 to include the timestamp prefix.");
	});

	it("should generate an ID of the specified length", () => {
		const length = 15;
		const id = generateRandomId(length);
		expect(id).toHaveLength(length);
	});

	it("should generate an ID with a timestamp prefix of length 8", () => {
		const id = generateRandomId(12);
		expect(id).toHaveProperty("substring");
		expect(id).toHaveProperty("length");
		expect(id.length).toBeGreaterThanOrEqual(8);
	});

	it("should generate an ID containing only allowed characters", () => {
		const allowedChars = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
		const id = generateRandomId(20);
		for (const char of id) {
			expect(allowedChars).toContain(char);
		}
	});

	it("should generate IDs that are sortable by creation time", () => {
		const id1 = generateRandomId(10);
		// Simulate a slight delay to ensure different timestamps
		return new Promise((resolve) => {
			setTimeout(() => {
				const id2 = generateRandomId(10);
				expect(id2 >= id1).toBe(true);
				resolve(null);
			}, 10);
		});
	});
});

describe("decodeTimestampFromId", () => {
	it("should throw an error if the ID length is less than 8", () => {
		expect(() => decodeTimestampFromId("abcdefg")).toThrowError(
			"ID must be at least 8 characters long to contain a timestamp prefix."
		);
	});

	it("should throw an error if the timestamp prefix contains invalid characters", () => {
		const invalidId = "0bcdefghijk";
		expect(() => decodeTimestampFromId(invalidId)).toThrowError('Invalid character "0" found in timestamp prefix.');
	});

	it("should correctly decode the timestamp from a generated ID", () => {
		const now = Date.now();
		// Mock Date.now to control the timestamp for testing
		const originalDateNow = Date.now;
		global.Date.now = () => now;
		const id = generateRandomId(16);
		global.Date.now = originalDateNow; // Restore original Date.now

		const decodedTimestamp = decodeTimestampFromId(id);
		// Allow a small margin for potential timing differences in test execution
		expect(decodedTimestamp).toBeGreaterThanOrEqual(now - 1);
		expect(decodedTimestamp).toBeLessThanOrEqual(now + 1);
	});

	it("should correctly decode the timestamp even with a longer ID", () => {
		const now = Date.now();
		const originalDateNow = Date.now;
		global.Date.now = () => now;
		const id = generateRandomId(20);
		global.Date.now = originalDateNow;

		const decodedTimestamp = decodeTimestampFromId(id);
		expect(decodedTimestamp).toBeGreaterThanOrEqual(now - 1);
		expect(decodedTimestamp).toBeLessThanOrEqual(now + 1);
	});
});
