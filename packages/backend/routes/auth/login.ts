import { createSession, getSessionExpirationDate, verifyUser } from "@backend/lib/auth";
import { Elysia, t } from "elysia";
import { ip } from "elysia-ip";

export const loginHandler = new Elysia({ prefix: "/auth" }).use(ip()).post(
	"/session",
	async ({ body, set, cookie, ip, request }) => {
		const { username, password } = body;

		const user = await verifyUser(username, password);
		if (!user) {
			set.status = 401;
			return { message: "Invalid credentials." };
		}

		const userAgent = request.headers.get("user-agent") || "Unknown";
		const sessionId = await createSession(user.id, ip || null, userAgent);

		const expiresAt = getSessionExpirationDate(365);
		cookie.sessionId.value = sessionId;
		cookie.sessionId.httpOnly = true;
		cookie.sessionId.secure = process.env.NODE_ENV === "production";
		cookie.sessionId.sameSite = "strict";
		cookie.sessionId.expires = expiresAt;

		return {
			message: "You are logged in.",
			sessionID: sessionId,
			user: {
				id: user.id,
				nickname: user.nickname,
				role: user.role,
				username: user.username,
			},
		};
	},
	{
		body: t.Object({
			password: t.String(),
			username: t.String(),
		}),
		detail: {
			description:
				"This endpoint authenticates users by verifying their credentials and creates a new session. \
			Upon successful authentication, it returns user information and sets a secure HTTP-only cookie \
			for session management. The session includes IP address and user agent tracking for security purposes.",
			summary: "User login",
		},
		response: {
			200: t.Object({
				message: t.String(),
				sessionID: t.String(),
				user: t.Object({
					id: t.Integer(),
					nickname: t.Optional(t.String()),
					role: t.String(),
					username: t.String(),
				}),
			}),
			401: t.Object({
				message: t.String(),
			}),
		},
	}
);
