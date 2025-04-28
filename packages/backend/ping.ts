import { createHandlers } from "./utils.ts";

export const pingHandler = createHandlers(async (c) => {
	return c.json({
		"message": "pong"
	});
});