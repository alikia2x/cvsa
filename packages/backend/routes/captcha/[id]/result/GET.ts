import { Context } from "hono";
import { Bindings, BlankEnv } from "hono/types";
import { ErrorResponse } from "src/schema";
import { createHandlers } from "src/utils.ts";
import { sign } from 'hono/jwt'
import { generateRandomId } from "@core/lib/randomID.ts";
import { getJWTsecret } from "lib/auth/getJWTsecret.ts";

interface CaptchaResponse {
	success: boolean;
	difficulty?: number;
	error?: string;
}

const getChallengeVerificationResult = async (id: string, ans: string) => {
	const baseURL = process.env["UCAPTCHA_URL"];
	const url = new URL(baseURL);
	url.pathname = `/challenge/${id}/validation`;
	return await fetch(url.toString(), {
		method: "POST",
		headers: {
			"Content-Type": "application/json"
		},
		body: JSON.stringify({
			y: ans
		})
	});
};

export const verifyChallengeHandler = createHandlers(
	async (c: Context<BlankEnv & { Bindings: Bindings }, "/captcha/:id/result">) => {
		const id = c.req.param("id");
		const ans = c.req.query("ans");
		if (!ans) {
			const response: ErrorResponse = {
				message: "Missing required query parameter: ans",
				code: "INVALID_QUERY_PARAMS"
			};
			return c.json<ErrorResponse>(response, 400);
		}
		const res = await getChallengeVerificationResult(id, ans);
		const data: CaptchaResponse = await res.json();
		if (data.error && res.status === 404) {
			const response: ErrorResponse = {
				message: data.error,
				code: "ENTITY_NOT_FOUND"
			};
			return c.json<ErrorResponse>(response, 401);
		} else if (data.error && res.status === 400) {
			const response: ErrorResponse = {
				message: data.error,
				code: "INVALID_QUERY_PARAMS"
			};
			return c.json<ErrorResponse>(response, 400);
		} else if (data.error) {
			const response: ErrorResponse = {
				message: data.error,
				code: "UNKNOWN_ERROR"
			};
			return c.json<ErrorResponse>(response, 500);
		}
		if (!data.success) {
			const response: ErrorResponse = {
				message: "Incorrect answer",
				code: "INVALID_CREDENTIALS"
			};
			return c.json<ErrorResponse>(response, 401);
		}

		const [r, err] = getJWTsecret();
		if (err) {
			return c.json<ErrorResponse>(r as ErrorResponse, 500);
		}
		const jwtSecret = r as string;

		const tokenID = generateRandomId(6);
		const NOW = Math.floor(Date.now() / 1000)
		const FIVE_MINUTES_LATER = NOW + 60 * 5;
		const jwt = await sign({
			difficulty: data.difficulty!,
			id: tokenID,
			exp: FIVE_MINUTES_LATER
		}, jwtSecret);
		return c.json({
			token: jwt
		});
	}
);
