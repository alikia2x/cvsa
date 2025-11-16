import Elysia from "elysia";

const encoder = new TextEncoder();

export const onAfterHandler = new Elysia().onAfterHandle(
	{ as: "global" },
	({ responseValue, request }) => {
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
	}
);
