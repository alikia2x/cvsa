import type { Context } from "hono";
import { createHandlers } from "./utils.ts";
import type { BlankEnv, BlankInput } from "hono/types";
import { getVideoSnapshots, getVideoSnapshotsByBV } from "@core/db/videoSnapshot.ts";
import type { VideoSnapshotType } from "@core/db/schema.d.ts";
import { boolean, mixed, number, object, ValidationError } from "yup";

const SnapshotQueryParamsSchema = object({
	ps: number().optional().positive(),
	pn: number().optional().positive(),
	offset: number().optional().positive(),
	reverse: boolean().optional(),
});

const idSchema = mixed().test(
	"is-valid-id",
	'id must be a string starting with "av" followed by digits, or "BV" followed by 10 alphanumeric characters, or a positive integer',
	async (value) => {
		if (value && await number().isValid(value)) {
			return Number.isInteger(value) && (value as number) > 0;
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
	},
);

type ContextType = Context<BlankEnv, "/video/:id/snapshots", BlankInput>;
export const getSnapshotsHanlder = createHandlers(async (c: ContextType) => {
	const client = c.get("db");

	try {
		const idParam = await idSchema.validate(c.req.param("id"));
		let videoId: number | string = idParam as string | number;
		if (typeof videoId === "string" && videoId.startsWith("av")) {
			videoId = videoId.slice(2);
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
			result = await getVideoSnapshots(client, videoId, limit, pageOrOffset, reverse, mode);
		} else {
			result = await getVideoSnapshotsByBV(client, videoId, limit, pageOrOffset, reverse, mode);
		}

		const rows = result.map((row) => ({
			...row,
			aid: Number(row.aid),
		}));

		return c.json(rows);
	} catch (e) {
		if (e instanceof ValidationError) {
			return c.json({ message: "Invalid query parameters", errors: e.errors }, 400);
		} else {
			return c.json({ message: "Unhandled error", error: e }, 500);
		}
	}
});
