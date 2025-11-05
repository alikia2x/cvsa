import { dbCred } from "@core/drizzle";
import { users } from "@core/drizzle/cred/schema";
import { Elysia, t } from "elysia";

export const loginHandler = new Elysia({ prefix: "/login" }).post(
	"/session",
	async ({ params, status, body }) => {
		const { username, password } = body;
		
		return {};
	},
	{
		response: {
			200: t.Object({}),
			404: t.Object({
				message: t.String()
			})
		},
		body: t.Object({
			username: t.String(),
			password: t.String()
		})
	}
);
