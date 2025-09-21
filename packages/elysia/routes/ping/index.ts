import { VERESION } from "@elysia/src";
import { Elysia, t } from "elysia";

export const pingHandler = new Elysia({ prefix: "/ping" }).get(
	"/",
	async (c) => {
		return {
			message: "pong",
			request: {
				headers: c.headers,
				ip: c.server?.requestIP(c.request)?.address,
				method: c.request.method,
				body: c.body,
				url: c.request.url
			},
			response: {
				time: new Date().getTime(),
				status: 200,
				version: VERESION
			}
		};
	},
	{
		response: {
			200: t.Object({
				message: t.String(),
				request: t.Object({
					headers: t.Any(),
					ip: t.Optional(t.String()),
					method: t.String(),
					body: t.Optional(t.Union([t.String(), t.Null()])),
					url: t.String()
				}),
				response: t.Object({
					time: t.Number(),
					status: t.Number(),
					version: t.String()
				})
			})
		},
		body: t.Optional(t.String()),
		detail: {
			summary: "Send a ping",
			description:
				"This endpoint returns a 'pong' message along with comprehensive information about the incoming request and the server's current status, including request headers, IP address, and server version. It's useful for monitoring API availability and debugging."
		}
	}
);
