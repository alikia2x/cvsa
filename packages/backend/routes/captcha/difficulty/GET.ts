import { createHandlers } from "src/utils";
import { object, string, ValidationError } from "yup";
import { ErrorResponse } from "src/schema";
import { getCurrentCaptchaDifficulty } from "@/lib/auth/captchaDifficulty";
import { sqlCred } from "@core/db/dbNew";

const queryParamsSchema = object({
	route: string().matches(/(?:GET|POST|PUT|PATCH|DELETE)-\/.*/g)
});

export const getCaptchaDifficultyHandler = createHandlers(async (c) => {
	try {
		const queryParams = await queryParamsSchema.validate(c.req.query());
		const { route } = queryParams;
		const difficulty = await getCurrentCaptchaDifficulty(sqlCred, route);
		if (!difficulty) {
			const response: ErrorResponse<unknown> = {
				code: "ENTITY_NOT_FOUND",
				message: "No difficulty configs found for this route.",
				errors: []
			};
			return c.json<ErrorResponse<unknown>>(response, 404);
		}
		return c.json({
			difficulty: difficulty
		});
	} catch (e: unknown) {
		if (e instanceof ValidationError) {
			const response: ErrorResponse = {
				code: "INVALID_QUERY_PARAMS",
				message: "Invalid query parameters",
				errors: e.errors
			};
			return c.json<ErrorResponse>(response, 400);
		} else {
			const response: ErrorResponse<unknown> = {
				code: "UNKNOWN_ERROR",
				message: "Unknown error",
				errors: [e]
			};
			return c.json<ErrorResponse<unknown>>(response, 500);
		}
	}
});
