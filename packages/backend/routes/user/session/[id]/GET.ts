import { Context } from "hono";
import { Bindings, BlankEnv } from "hono/types";
import { ErrorResponse } from "src/schema";
import { createHandlers } from "src/utils";
import { sqlCred } from "@core/db/dbNew";
import { UserType } from "@core/db/schema";

export const getUserByLoginSessionHandler = createHandlers(
	async (c: Context<BlankEnv & { Bindings: Bindings }, "/user/session/:id">) => {
		const id = c.req.param("id");
		const users = await sqlCred<UserType[]>`
    		SELECT u.*
            FROM users u
            JOIN login_sessions ls ON u.id = ls.uid
            WHERE ls.id = ${id};
        `;
		if (users.length === 0) {
			const response: ErrorResponse = {
				message: "Cannot find user",
				code: "ENTITY_NOT_FOUND",
				errors: []
			};
			return c.json<ErrorResponse>(response, 404);
		}
		const user = users[0];
		return c.json({
			username: user.username,
			nickname: user.nickname,
			role: user.role
		});
	}
);
