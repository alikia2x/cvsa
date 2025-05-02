import { bodyLimit } from "hono/body-limit";
import { ErrorResponse } from "../src/schema";

export const bodyLimitForPing = bodyLimit({
	maxSize: 14000,
	onError: (c) => {
		const res: ErrorResponse<string> = {
			message: "Body too large",
			errors: ["Body should not be larger than 14kB."],
			code: "BODY_TOO_LARGE"
		};
		return c.json(res, 413);
	}
});
