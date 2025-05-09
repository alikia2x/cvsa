import { Context } from "hono";

export const notFoundRoute = (c: Context) => {
	return c.json(
		{
			message: "Not Found"
		},
		404
	);
};