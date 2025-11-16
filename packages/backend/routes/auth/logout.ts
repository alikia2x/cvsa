import { Elysia, t } from "elysia";
import { deactivateSession } from "@backend/lib/auth";
import requireAuth from "@backend/middlewares/auth";

export const logoutHandler = new Elysia({ prefix: "/auth" }).use(requireAuth).delete(
	"/session",
	async ({ set, session, cookie }) => {
		const sessionId = session.sessionId;

		if (!sessionId) {
			set.status = 401;
			return { message: "Not authenticated." };
		}

		await deactivateSession(sessionId as string);
		cookie.sessionId.remove();

		return { message: "Successfully logged out." };
	},
	{
		response: {
			200: t.Object({
				message: t.String()
			}),
			401: t.Object({
				message: t.String()
			})
		},
		detail: {
			summary: "Logout current session",
			description:
				"This endpoint logs out the current user by deactivating their session and removing the session cookie. \
				It requires an active session cookie to be present in the request. After successful logout, the session \
				is invalidated and cannot be used again."
		}
	}
);
