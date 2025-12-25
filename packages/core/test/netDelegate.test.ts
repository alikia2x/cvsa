import { describe, expect, test } from "bun:test";
import networkDelegate from "@core/net/delegate";

describe("proxying requests", () => {
	test("Alibaba Cloud FC", async () => {
		const { data } = await networkDelegate.request<{
			headers: Record<string, string>;
		}>("https://postman-echo.com/get", "test");
		expect(data.headers.referer).toBe("https://www.bilibili.com/");
	});
	test("CF Proxy", async () => {
		const { data } = await networkDelegate.request<{
			headers: Record<string, string>;
		}>("https://postman-echo.com/get", "testCf");
		expect(data.headers).toBeObject();
	});
});
