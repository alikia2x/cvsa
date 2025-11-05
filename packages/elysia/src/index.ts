import { Elysia } from "elysia";
import { getBindingInfo, logStartup } from "./startMessage";
import { pingHandler } from "@elysia/routes/ping";
import openapi from "@elysiajs/openapi";
import { cors } from "@elysiajs/cors";
import { songInfoHandler } from "@elysia/routes/song/info";
import { rootHandler } from "@elysia/routes/root";
import { getVideoMetadataHandler } from "@elysia/routes/video/metadata";
import { closeMileStoneHandler } from "@elysia/routes/song/milestone";

const [host, port] = getBindingInfo();
logStartup(host, port);

const encoder = new TextEncoder();

const app = new Elysia({
	serve: {
		hostname: host
	}
})
	.onAfterHandle({ as: "global" }, ({ responseValue, request }) => {
		const contentType = request.headers.get("Content-Type") || "";
		const accept = request.headers.get("Accept") || "";
		const secFetchMode = request.headers.get("Sec-Fetch-Mode");
		const requestJson = contentType.includes("application/json");
		const isBrowser =
			!requestJson && (accept.includes("text/html") || secFetchMode === "navigate");
		const responseValueType = typeof responseValue;
		const isObject = responseValueType === "object";
		if (!isObject) {
			const response = {
				message: responseValue
			};
			const text = isBrowser ? JSON.stringify(response, null, 2) : JSON.stringify(response);
			return new Response(encoder.encode(text), {
				headers: {
					"Content-Type": "application/json; charset=utf-8"
				}
			});
		}
		const realResponse = responseValue as Record<string, unknown>;
		if (realResponse.code) {
			const text = isBrowser
				? JSON.stringify(realResponse.response, null, 2)
				: JSON.stringify(realResponse.response);
			return new Response(encoder.encode(text), {
				status: realResponse.code as any,
				headers: {
					"Content-Type": "application/json; charset=utf-8"
				}
			});
		}
		const text = isBrowser
			? JSON.stringify(realResponse, null, 2)
			: JSON.stringify(realResponse);
		return new Response(encoder.encode(text), {
			headers: {
				"Content-Type": "application/json; charset=utf-8"
			}
		});
	})
	.use(cors())
	.use(openapi())
	.use(rootHandler)
	.use(pingHandler)
	.use(getVideoMetadataHandler)
	.use(songInfoHandler)
	.use(closeMileStoneHandler)
	.listen(15412);

export const VERSION = "0.7.0";

export type App = typeof app;
