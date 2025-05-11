import { Context } from "hono";
import { Bindings, BlankEnv, BlankInput } from "hono/types";
import { ErrorResponse } from "src/schema";
import { createHandlers } from "src/utils.ts";
import * as jose from "jose";

interface CaptchaResponse {
	success: boolean;
	error?: string;
}

const getChallengeVerificationResult = async (id: string, ans: string) => {
	const baseURL = process.env["UCAPTCHA_URL"];
	const url = new URL(baseURL);
	url.pathname = `/challenge/${id}/validation`;
	const res = await fetch(url.toString(), {
		method: "POST",
		headers: {
			"Content-Type": "application/json"
		},
		body: JSON.stringify({
			y: ans
		})
	});
	return res;
};

export const verifyChallengeHandler = createHandlers(
	async (c: Context<BlankEnv & { Bindings: Bindings }, "/captcha/:id/result", BlankInput>) => {
		const id = c.req.param("id");
		const ans = c.req.query("ans");
		if (!ans) {
			const response: ErrorResponse<string> = {
				message: "Missing required query parameter: ans",
				code: "INVALID_QUERY_PARAMS"
			};
			return c.json<ErrorResponse<string>>(response, 400);
		}
		const res = await getChallengeVerificationResult(id, ans);
		const data: CaptchaResponse = await res.json();
		if (data.error && res.status === 404) {
			const response: ErrorResponse<string> = {
				message: data.error,
				code: "ENTITY_NOT_FOUND"
			};
			return c.json<ErrorResponse<string>>(response, 401);
		} else if (data.error && res.status === 400) {
			const response: ErrorResponse<string> = {
				message: data.error,
				code: "INVALID_QUERY_PARAMS"
			};
			return c.json<ErrorResponse<string>>(response, 400);
		} else if (data.error) {
			const response: ErrorResponse<string> = {
				message: data.error,
				code: "UNKNOWN_ERROR"
			};
			return c.json<ErrorResponse<string>>(response, 500);
		}
		if (!data.success) {
			const response: ErrorResponse<string> = {
				message: "Incorrect answer",
				code: "INVALID_CREDENTIALS"
			};
			return c.json<ErrorResponse<string>>(response, 401);
		}

		const secret = process.env["JWT_SECRET"];
		if (!secret) {
			const response: ErrorResponse<string> = {
				message: "JWT_SECRET is not set",
				code: "SERVER_ERROR"
			};
			return c.json<ErrorResponse<string>>(response, 500);
		}
		const jwtSecret = new TextEncoder().encode(secret);
		const alg = "HS256";

		const jwt = await new jose.SignJWT()
			.setProtectedHeader({ alg })
			.setIssuedAt()
			.sign(jwtSecret);
		return c.json({
			token: jwt
		});
	}
);
