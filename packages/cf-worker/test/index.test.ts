import { createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { describe, expect, it } from "vitest";
import worker from "../src/index";

// For now, you'll need to do something like this to get a correctly-typed
// `Request` to pass to `worker.fetch()`.
const IncomingRequest = Request<unknown, IncomingRequestCfProperties>;

interface ProxyResponseBody {
	data: string;
	time: number;
	error?: string;
}

describe("Proxy Worker", () => {
	it("rejects non-POST requests", async () => {
		const request = new IncomingRequest("http://example.com", { method: "GET" });
		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.status).toBe(405);
		const body = (await response.json()) as ProxyResponseBody;
		expect(body.error).toBe("Method not allowed");
		expect(body.data).toBe("");
		expect(typeof body.time).toBe("number");
	});

	it("rejects Invalid request", async () => {
		const request = new IncomingRequest("http://example.com", {
			method: "POST",
			body: "invalid json",
			headers: { "Content-Type": "application/json" },
		});
		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.status).toBe(400);
		const body = (await response.json()) as ProxyResponseBody;
		expect(body.error).toBe("Invalid request");
		expect(body.data).toBe("");
		expect(typeof body.time).toBe("number");
	});

	it("rejects missing url field", async () => {
		const request = new IncomingRequest("http://example.com", {
			method: "POST",
			body: JSON.stringify({ headers: { "X-Custom": "value" } }),
			headers: { "Content-Type": "application/json" },
		});
		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.status).toBe(400);
		const body = (await response.json()) as ProxyResponseBody;
		expect(body.error).toBe("Invalid request");
		expect(body.data).toBe("");
		expect(typeof body.time).toBe("number");
	});

	it("rejects invalid URL format", async () => {
		const request = new IncomingRequest("http://example.com", {
			method: "POST",
			body: JSON.stringify({ url: "not-a-valid-url", headers: {} }),
			headers: { "Content-Type": "application/json" },
		});
		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.status).toBe(400);
		const body = (await response.json()) as ProxyResponseBody;
		expect(body.error).toBe("Invalid request");
		expect(body.data).toBe("");
		expect(typeof body.time).toBe("number");
	});

	it("includes CORS headers in response", async () => {
		const request = new IncomingRequest("http://example.com", {
			method: "POST",
			body: JSON.stringify({
				url: "https://httpbin.org/status/200",
				headers: {},
			}),
			headers: { "Content-Type": "application/json" },
		});

		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
		expect(response.headers.get("Content-Type")).toBe("application/json");
	});

	it("handles proxy request failures", async () => {
		const request = new IncomingRequest("http://example.com", {
			method: "POST",
			body: JSON.stringify({
				url: "https://invalid-domain-that-does-not-exist-12345.com",
				headers: {},
			}),
			headers: { "Content-Type": "application/json" },
		});

		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.status).toBe(504);
		const body = (await response.json()) as ProxyResponseBody;
		expect(body.error).toContain("Socket timeout");
		expect(body.data).toBe("");
		expect(typeof body.time).toBe("number");
	});

	it("returns JSON response with data and time fields on success", async () => {
		const request = new IncomingRequest("http://example.com", {
			method: "POST",
			body: JSON.stringify({
				url: "https://postman-echo.com/get",
				headers: {},
			}),
			headers: { "Content-Type": "application/json" },
		});

		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(response.status).toBe(200);
		const body = (await response.json()) as ProxyResponseBody;
		expect(typeof body.data).toBe("string");
		expect(body.data.length).toBeGreaterThan(0);
		expect(typeof body.time).toBe("number");
		expect(body.time).toBeLessThanOrEqual(Date.now());
		expect(body.time).toBeGreaterThan(Date.now() - 60000); // Within last minute
	});
});
