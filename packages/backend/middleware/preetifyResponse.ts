import { startTime, endTime } from "hono/timing";
import { Context, Next } from "hono";

export const preetifyResponse = async (c: Context, next: Next) => {
	await next();
	const contentType = c.res.headers.get("Content-Type") || "";
	if (!contentType.includes("application/json")) return;
	const accept = c.req.header("Accept") || "";
	const secFetchMode = c.req.header("Sec-Fetch-Mode");
	const isBrowser = accept.includes("text/html") || secFetchMode === "navigate";
	if (isBrowser) {
		const json = await c.res.json();
		startTime(c, "seralize", "Prettify the response");
		const prettyJson = JSON.stringify(json, null, 2);
		endTime(c, "seralize");
		c.res = new Response(prettyJson, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
	}
};