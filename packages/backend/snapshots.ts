import type { Context } from "hono";
import { createHandlers } from "./utils.ts";
import type { BlankEnv, BlankInput } from "hono/types";
import { getVideoSnapshots, getVideoSnapshotsByBV } from "./db/videoSnapshot.ts";
import type { VideoSnapshotType } from "@core/db/schema.d.ts";
import { boolean, mixed, number, object, ValidationError } from "yup";
import { ErrorResponse } from "./schema";

const SnapshotQueryParamsSchema = object({
	ps: number().integer().optional().positive(),
	pn: number().integer().optional().positive(),
	offset: number().integer().optional().positive(),
	reverse: boolean().optional()
});

export const idSchema = mixed().test(
	"is-valid-id",
	'id must be a string starting with "av" followed by digits, or "BV" followed by 10 alphanumeric characters, or a positive integer',
	async (value) => {
		if (value && (await number().integer().isValid(value))) {
			const v = parseInt(value as string);
			return Number.isInteger(v) && v > 0;
		}

		if (typeof value === "string") {
			if (value.startsWith("av")) {
				const digitsOnly = value.substring(2);
				return /^\d+$/.test(digitsOnly) && digitsOnly.length > 0;
			}

			if (value.startsWith("BV")) {
				const remainingChars = value.substring(2);
				return /^[a-zA-Z0-9]{10}$/.test(remainingChars);
			}
		}

		return false;
	}
);

type ContextType = Context<BlankEnv, "/video/:id/snapshots", BlankInput>;
export const getSnapshotsHanlder = createHandlers(async (c: ContextType) => {
	try {
		const idParam = await idSchema.validate(c.req.param("id"));
		let videoId: string | number = idParam as string;
		if (videoId.startsWith("av")) {
			videoId = parseInt(videoId.slice(2));
		} else if (await number().isValid(videoId)) {
			videoId = parseInt(videoId);
		}
		const queryParams = await SnapshotQueryParamsSchema.validate(c.req.query());
		const { ps, pn, offset, reverse = false } = queryParams;

		let limit = 1000;
		if (ps && ps > 1) {
			limit = ps;
		}

		let pageOrOffset = 1;
		let mode: "page" | "offset" = "page";

		if (pn && pn > 1) {
			pageOrOffset = pn;
			mode = "page";
		} else if (offset && offset > 1) {
			pageOrOffset = offset;
			mode = "offset";
		}

		let result: VideoSnapshotType[];

		if (typeof videoId === "number") {
			result = await getVideoSnapshots(videoId, limit, pageOrOffset, reverse, mode);
		} else {
			result = await getVideoSnapshotsByBV(videoId, limit, pageOrOffset, reverse, mode);
		}

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
				errors: e.errors,
			}
			return c.json<ErrorResponse<string>>(response, 400);
		} else {
			const response: ErrorResponse<unknown> = {
				code: "UNKNOWN_ERR",
				message: "Unhandled error",
				errors: [e]
			}
			return c.json<ErrorResponse<unknown>>(response, 500);
		}
	}
});
