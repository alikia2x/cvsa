import networkDelegate from "@core/net/delegate";
import { test, expect, describe } from "bun:test";

describe("proxying requests", () => {
	test("Alibaba Cloud FC", async () => {
		const { data } = (await networkDelegate.request<{
			headers: Record<string, string>;
		}>(
			"https://postman-echo.com/get",
			"test"
		));
        expect(data.headers.referer).toBe('https://www.bilibili.com/');
	});
	test("IP Proxy", async () => {
		const { data } = await networkDelegate.request<{
			headers: Record<string, string>;
		}>("https://postman-echo.com/get", "test_ip");
		expect(data.headers).toBeObject();
	});
});
