import requireAuth from "@backend/middlewares/auth";
import { Elysia, t } from "elysia";

export const getCurrentUserHandler = new Elysia().use(requireAuth).get(
	"/user",
	async ({ user, status }) => {
		if (!user) {
			return status(401, { message: "Unauthorized" });
		}
		return {
			id: user.id,
			username: user.username,
			nickname: user.nickname,
			role: user.role,
		};
	},
	{
		response: {
			200: t.Object({
				id: t.Integer(),
				username: t.String(),
				nickname: t.Union([t.String(), t.Null()]),
				role: t.String(),
			}),
			401: t.Object({
				message: t.String(),
			}),
		},
	}
);
