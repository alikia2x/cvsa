import { getClientIP } from "middleware/logger";
import { createHandlers } from "src/utils";
import { VERSION } from "src/main";

export const pingHandler = createHandlers(async (c) => {
	const requestHeaders = c.req.raw.headers;
	return c.json({
		message: "pong",
		request: {
			headers: requestHeaders,
			ip: getClientIP(c),
			mode: c.req.raw.mode,
			method: c.req.method,
			query: new URL(c.req.url).searchParams,
			body: await c.req.text(),
			url: c.req.raw.url
		},
		response: {
			time: new Date().getTime(),
			status: 200,
			version: VERSION
		}
	});
});
