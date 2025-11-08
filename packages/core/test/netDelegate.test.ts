import networkDelegate from "@core/net/delegate";
import { test, expect, describe } from "bun:test";

describe("proxying requests", () => {
	test("Alibaba Cloud FC", async () => {
		const { res } = await networkDelegate.request("https://postman-echo.com/get", "test") as any;
        expect(res.headers.referer).toBe('https://www.bilibili.com/');
	});
});
