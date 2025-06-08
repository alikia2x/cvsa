import { ApiRequestError, fetcher } from "@/lib/net";
import { ErrorResponse } from "@cvsa/backend";
import { cookies } from "next/headers";

export async function POST() {
	const backendURL = process.env.BACKEND_URL || "";
	const cookieStore = await cookies();
	const sessionID = cookieStore.get("session_id");
	if (!sessionID) {
		const response: ErrorResponse<string> = {
			message: "No session_id provided",
			errors: [],
			code: "ENTITY_NOT_FOUND"
		};
		return new Response(JSON.stringify(response), {
			status: 401
		});
	}

	try {
		const response = await fetcher(`${backendURL}/session/${sessionID.value}`, {
			method: "DELETE"
		});

		const headers = response.headers;

		return new Response(null, {
			status: 204,
			headers: {
				"Set-Cookie": (headers["set-cookie"] || [""])[0]
			}
		});
	} catch (error) {
		if (error instanceof ApiRequestError && error.response) {
			const res = error.response;
			const code = error.code;
			return new Response(JSON.stringify(res), {
				status: code
			});
		} else if (error instanceof Error) {
			const response: ErrorResponse<string> = {
				message: error.message,
				errors: [],
				code: "SERVER_ERROR"
			};
			return new Response(JSON.stringify(response), {
				status: 500
			});
		} else {
			const response: ErrorResponse<string> = {
				message: "Unknown error occurred",
				errors: [],
				code: "UNKNOWN_ERROR"
			};
			return new Response(JSON.stringify(response), {
				status: 500
			});
		}
	}
}
