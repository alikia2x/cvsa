import type { Context } from "hono";
import { createHandlers } from "src/utils";
import type { BlankEnv, BlankInput } from "hono/types";
import { number, object, ValidationError } from "yup";
import { ErrorResponse } from "src/schema";
import { startTime, endTime } from "hono/timing";
import { getVideosInViewsRange } from "@/db/latestSnapshots";

const SnapshotQueryParamsSchema = object({
	min_views: number().integer().optional().positive(),
	max_views: number().integer().optional().positive()
});

type ContextType = Context<BlankEnv, "/videos", BlankInput>;

export const getVideosHanlder = createHandlers(async (c: ContextType) => {
	startTime(c, "parse", "Parse the request");
	try {
		const queryParams = await SnapshotQueryParamsSchema.validate(c.req.query());
		const { min_views, max_views } = queryParams;

		if (!min_views && !max_views) {
			const response: ErrorResponse<string> = {
				code: "INVALID_QUERY_PARAMS",
				message: "Invalid query parameters",
				errors: ["Must provide one of these query parameters: min_views, max_views"]
			};
			return c.json<ErrorResponse<string>>(response, 400);
		}

		endTime(c, "parse");

		startTime(c, "db", "Query the database");

		const minViews = min_views ? min_views : 0;
		const maxViews = max_views ? max_views : 2147483647;

		const result = await getVideosInViewsRange(minViews, maxViews);

		endTime(c, "db");

		const rows = result.map((row) => ({
			...row,
			aid: Number(row.aid)
		}));

		return c.json(rows);
	} catch (e: unknown) {
		if (e instanceof ValidationError) {
			const response: ErrorResponse<string> = {
				code: "INVALID_QUERY_PARAMS",
				message: "Invalid query parameters",
				errors: e.errors
			};
			return c.json<ErrorResponse<string>>(response, 400);
		} else {
			const response: ErrorResponse<unknown> = {
				code: "UNKNOWN_ERROR",
				message: "Unhandled error",
				errors: [e]
			};
			return c.json<ErrorResponse<unknown>>(response, 500);
		}
	}
});
