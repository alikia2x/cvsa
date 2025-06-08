import { Context } from "hono";
import { Bindings, BlankEnv } from "hono/types";
import { ErrorResponse } from "src/schema";
import { createHandlers } from "src/utils.ts";
import { sqlCred } from "@core/db/dbNew";
import { object, string, ValidationError } from "yup";
import { setCookie } from "hono/cookie";

const loginSessionExists = async (sessionID: string) => {
	const result = await sqlCred`
        SELECT 1
        FROM login_sessions
        WHERE id = ${sessionID}
	`;
	return result.length > 0;
};

export const logoutHandler = createHandlers(async (c: Context<BlankEnv & { Bindings: Bindings }, "/session/:id">) => {
	try {
		const session_id = c.req.param("id");

		const exists = loginSessionExists(session_id);

		if (!exists) {
			const response: ErrorResponse<string> = {
				message: "Cannot found given session_id.",
				errors: [`Session ${session_id} not found`],
				code: "ENTITY_NOT_FOUND"
			};
			return c.json<ErrorResponse<string>>(response, 404);
		}

		await sqlCred`
		    UPDATE login_sessions
			SET deactivated_at = CURRENT_TIMESTAMP
			WHERE id = ${session_id}
		`;

		const isDev = process.env.NODE_ENV === "development";

		setCookie(c, "session_id", "", {
			path: "/",
			maxAge: 0,
			domain: process.env.DOMAIN,
			secure: isDev ? true : true,
			sameSite: isDev ? "None" : "Lax",
			httpOnly: true
		});

		return c.body(null, 204);
	} catch (e) {
		if (e instanceof ValidationError) {
			const response: ErrorResponse<string> = {
				message: "Invalid registration data.",
				errors: e.errors,
				code: "INVALID_PAYLOAD"
			};
			return c.json<ErrorResponse<string>>(response, 400);
		} else if (e instanceof SyntaxError) {
			const response: ErrorResponse<string> = {
				message: "Invalid JSON payload.",
				errors: [e.message],
				code: "INVALID_FORMAT"
			};
			return c.json<ErrorResponse<string>>(response, 400);
		} else {
			const response: ErrorResponse<string> = {
				message: "Unknown error.",
				errors: [(e as Error).message],
				code: "UNKNOWN_ERROR"
			};
			return c.json<ErrorResponse<string>>(response, 500);
		}
	}
});
