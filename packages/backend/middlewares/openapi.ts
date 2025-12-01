import openapi from "@elysiajs/openapi";
import pkg from "../package.json";
import * as z from "zod";
import { fromTypes } from "@elysiajs/openapi";

export const openAPIMiddleware = openapi({
	documentation: {
		info: {
			title: "CVSA API Docs",
			version: pkg.version
		}
	},
	mapJsonSchema: {
		zod: z.toJSONSchema
	},
	references: fromTypes(),
	scalar: {
		theme: "kepler",
		hideClientButton: true,
		hideDarkModeToggle: true
	}
});
