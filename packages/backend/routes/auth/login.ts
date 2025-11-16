import { Elysia, t } from "elysia";
import { ip } from "elysia-ip";
import { verifyUser, createSession, getSessionExpirationDate } from "@backend/lib/auth";

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

		const expiresAt = getSessionExpirationDate();
		cookie.sessionId.value = sessionId;
		cookie.sessionId.httpOnly = true;
		cookie.sessionId.secure = process.env.NODE_ENV === "production";
		cookie.sessionId.sameSite = "strict";
		cookie.sessionId.expires = expiresAt;

		return {
			message: "You are logged in.",
			user: {
				id: user.id,
				username: user.username,
				nickname: user.nickname,
				role: user.role
			},
			sessionID: sessionId
		};
	},
	{
		response: {
			200: t.Object({
				message: t.String(),
				user: t.Object({
					id: t.Integer(),
					username: t.String(),
					nickname: t.Optional(t.String()),
					role: t.String()
				}),
				sessionID: t.String()
			}),
			401: t.Object({
				message: t.String()
			})
		},
		body: t.Object({
			username: t.String(),
			password: t.String()
		})
	}
);
