import { Elysia } from "elysia";
import { jwt } from "@elysiajs/jwt";
import { redis } from "@core/db/redis";

interface JWTPayload {
	id: string;
	[key: string]: any;
}

export const captchaMiddleware = new Elysia({ name: "captcha" })
	.use(
		jwt({
			name: "captchaJwt",
			secret: process.env.JWT_SECRET || "default-secret-key"
		})
	)
	.derive(async ({ request, captchaJwt, set }) => {
		const authHeader = request.headers.get("authorization");

		if (!authHeader || !authHeader.startsWith("Bearer ")) {
			set.status = 401;
			throw new Error("Missing or invalid authorization header");
		}

		const token = authHeader.slice(7);
		try {
			const payload = (await captchaJwt.verify(token)) as JWTPayload;

			if (!payload || !payload.id) {
				set.status = 401;
				throw new Error("Invalid JWT payload");
			}

			const redisKey = `captcha:${payload.id}`;

			const exists = await redis.exists(redisKey);

			if (exists) {
				set.status = 400;
				throw new Error("Captcha already used or expired");
			}

			await redis.setex(redisKey, 300, "used");

			return {
				captchaVerified: true,
				userId: payload.id
			};
		} catch (error) {
			if (error instanceof Error) {
				set.status = 401;
				throw new Error(`JWT verification failed: ${error.message}`);
			}
			set.status = 500;
			throw new Error("Internal server error during captcha verification");
		}
	});

export default captchaMiddleware;
