import { VERSION } from "@backend/src";
import { Elysia, t } from "elysia";
import { ip } from "elysia-ip";

export const pingHandler = new Elysia({ prefix: "/ping" }).use(ip()).get(
	"/",
	async ({ headers, request, body, ip }) => {
		return {
			message: "pong",
			request: {
				body: body,
				headers: headers,
				ip: ip,
				method: request.method,
				url: request.url,
			},
			response: {
				status: 200,
				time: Date.now(),
				version: VERSION,
			},
		};
	},
	{
		body: t.Optional(t.String()),
		detail: {
			description:
				"This endpoint returns a 'pong' message along with comprehensive information about the incoming request and the server's current status, including request headers, IP address, and server version. It's useful for monitoring API availability and debugging.",
			summary: "Send a ping",
		},
		response: {
			200: t.Object({
				message: t.String(),
				request: t.Object({
					body: t.Optional(t.Union([t.String(), t.Null()])),
					headers: t.Any(),
					ip: t.Optional(t.String()),
					method: t.String(),
					url: t.String(),
				}),
				response: t.Object({
					status: t.Number(),
					time: t.Number(),
					version: t.String(),
				}),
			}),
		},
	}
);
