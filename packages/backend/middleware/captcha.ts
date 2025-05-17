import { Context, Next } from "hono";
import { ErrorResponse } from "src/schema";
import { SlidingWindow } from "@core/mq/slidingWindow.ts";
import { getCaptchaConfigMaxDuration, getCurrentCaptchaDifficulty } from "@/lib/auth/captchaDifficulty.ts";
import { sqlCred } from "@core/db/dbNew.ts";
import { redis } from "@core/db/redis.ts";
import { verify } from 'hono/jwt';
import { JwtTokenInvalid, JwtTokenExpired } from "hono/utils/jwt/types";
import { getJWTsecret } from "@/lib/auth/getJWTsecret.ts";
import { lockManager } from "@core/mq/lockManager.ts";
import { object, string, number, ValidationError } from "yup";

const tokenSchema = object({
	exp: number().integer(),
	id: string().length(6),
	difficulty: number().integer().moreThan(0)
});

export const captchaMiddleware = async (c: Context, next: Next) => {
	const authHeader = c.req.header("Authorization");

	if (!authHeader) {
		const response: ErrorResponse = {
			message: "'Authorization' header is missing.",
			code: "UNAUTHORIZED"
		};
		return c.json<ErrorResponse>(response, 401);
	}

	const authIsBearer = authHeader.startsWith("Bearer ");
	if (!authIsBearer || authHeader.length < 8) {
		const response: ErrorResponse = {
			message: "'Authorization' header is invalid.",
			code: "INVALID_HEADER"
		};
		return c.json<ErrorResponse>(response, 400);
	}

	const [r, err] = getJWTsecret();
	if (err) {
		return c.json<ErrorResponse>(r as ErrorResponse, 500);
	}
	const jwtSecret = r as string;

	const token = authHeader.substring(7);

	const path = c.req.path;
	const method = c.req.method;
	const route = `${method}-${path}`;

	const requiredDifficulty = await getCurrentCaptchaDifficulty(sqlCred, route);

	try {
		const decodedPayload = await verify(token, jwtSecret);
		const payload = await tokenSchema.validate(decodedPayload);
		const difficulty = payload.difficulty;
		const tokenID = payload.id;
		const consumed = await lockManager.isLocked(tokenID);
		if (consumed) {
			const response: ErrorResponse = {
				message: "Token has already been used.",
				code: "INVALID_CREDENTIALS"
			};
			return c.json<ErrorResponse>(response, 401);
		}
		if (difficulty < requiredDifficulty) {
			const response: ErrorResponse = {
				message: "Token to weak.",
				code: "UNAUTHORIZED"
			};
			return c.json<ErrorResponse>(response, 401);
		}
		const EXPIRE_FIVE_MINUTES = 300;
		await lockManager.acquireLock(tokenID, EXPIRE_FIVE_MINUTES);
	}
	catch (e) {
		if (e instanceof JwtTokenInvalid) {
			const response: ErrorResponse = {
				message: "Failed to verify the token.",
				code: "INVALID_CREDENTIALS"
			};
			return c.json<ErrorResponse>(response, 400);
		}
		else if (e instanceof JwtTokenExpired) {
			const response: ErrorResponse = {
				message: "Token expired.",
				code: "INVALID_CREDENTIALS"
			};
			return c.json<ErrorResponse>(response, 400);
		}
		else if (e instanceof ValidationError) {
			const response: ErrorResponse = {
				code: "INVALID_QUERY_PARAMS",
				message: "Invalid query parameters",
				errors: e.errors
			};
			return c.json<ErrorResponse>(response, 400);
		}
		else {
			const response: ErrorResponse = {
				message: "Unknown error.",
				code: "UNKNOWN_ERROR"
			};
			return c.json<ErrorResponse>(response, 500);
		}
	}
	const duration = await getCaptchaConfigMaxDuration(sqlCred, route);
	const window = new SlidingWindow(redis, duration);
	await window.event(`captcha-${route}`);

	await next();
};