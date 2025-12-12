import { VERSION } from "@backend/src";
import { Elysia, t } from "elysia";
import { ip } from "elysia-ip";

export const pingHandler = new Elysia({ prefix: "/ping" }).use(ip()).get(
	"/",
	async ({ headers, request, body, ip }) => {
		return {
			message: "pong",
			request: {
				headers: headers,
				ip: ip,
				method: request.method,
				body: body,
				url: request.url
			},
			response: {
				time: new Date().getTime(),
				status: 200,
				version: VERSION
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
