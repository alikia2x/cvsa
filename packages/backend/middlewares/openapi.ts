import openapi, { fromTypes } from "@elysiajs/openapi";
import * as z from "zod";
import pkg from "../package.json";

export const openAPIMiddleware = openapi({
	documentation: {
		info: {
			title: "CVSA API Docs",
			version: pkg.version,
		},
	},
	mapJsonSchema: {
		zod: z.toJSONSchema,
	},
	references: fromTypes(),
	scalar: {
		hideClientButton: true,
		hideDarkModeToggle: true,
		theme: "kepler",
	},
});
