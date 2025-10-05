import { createHandlers } from "src/utils";
import { getCurrentCaptchaDifficulty } from "@/lib/auth/captchaDifficulty";
import { sqlCred } from "@core/db/dbNew";
import { object, string, ValidationError } from "yup";
import { CaptchaSessionResponse, ErrorResponse } from "@/src/schema";
import type { ContentfulStatusCode } from "hono/utils/http-status";

const bodySchema = object({
	route: string().matches(/(?:GET|POST|PUT|PATCH|DELETE)-\/.*/g)
});

const createNewChallenge = async (difficulty: number) => {
	const baseURL = process.env["UCAPTCHA_URL"];
	const url = new URL(baseURL);
	url.pathname = "/challenge";
	return await fetch(url.toString(), {
		method: "POST",
		headers: {
			"Content-Type": "application/json"
		},
		body: JSON.stringify({
			difficulty: difficulty
		})
	});
};

export const createCaptchaSessionHandler = createHandlers(async (c) => {
	try {
		const requestBody = await bodySchema.validate(await c.req.json());
		const { route } = requestBody;
		const difficuly = await getCurrentCaptchaDifficulty(sqlCred, route);
		const res = await createNewChallenge(difficuly);
		return c.json<CaptchaSessionResponse | unknown>(await res.json(), res.status as ContentfulStatusCode);
	} catch (e: unknown) {
		if (e instanceof ValidationError) {
			const response: ErrorResponse = {
				code: "INVALID_QUERY_PARAMS",
				message: "Invalid query parameters",
				errors: e.errors
			};
			return c.json<ErrorResponse>(response, 400);
		} else {
			const response: ErrorResponse<unknown> = {
				code: "UNKNOWN_ERROR",
				message: "Unknown error",
				errors: [e]
			};
			return c.json<ErrorResponse<unknown>>(response, 500);
		}
	}
});
