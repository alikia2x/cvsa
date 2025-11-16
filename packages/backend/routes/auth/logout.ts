import { Elysia, t } from "elysia";
import { deactivateSession } from "@elysia/lib/auth";

export const logoutHandler = new Elysia({ prefix: "/auth" })
	.delete(
		"/session",
		async ({ set, cookie }) => {
			const sessionId = cookie.sessionId?.value;

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
			}
		}
	)
	.delete(
		"/session/:id",
		async ({ params }) => {
			const sessionId = params.id;

			await deactivateSession(sessionId as string);

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
			}
		}
	);
