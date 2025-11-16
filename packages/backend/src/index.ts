import { Elysia, file } from "elysia";
import { getBindingInfo, logStartup } from "./startMessage";
import { pingHandler } from "@backend/routes/ping";
import openapi from "@elysiajs/openapi";
import { cors } from "@elysiajs/cors";
import { songInfoHandler } from "@backend/routes/song/info";
import { rootHandler } from "@backend/routes/root";
import { getVideoMetadataHandler } from "@backend/routes/video/metadata";
import { closeMileStoneHandler } from "@backend/routes/song/milestone";
import { authHandler } from "@backend/routes/auth";
import { onAfterHandler } from "./onAfterHandle";
import { searchHandler } from "@backend/routes/search";
import { getVideoSnapshotsHandler } from "@backend/routes/video/snapshots";
import { addSongHandler } from "@backend/routes/song/add";
import { deleteSongHandler } from "@backend/routes/song/delete";
import { songEtaHandler } from "@backend/routes/video/eta";
import "./mq";
import pkg from "../package.json";
import * as z from "zod";
import { fromTypes } from "@elysiajs/openapi";

const [host, port] = getBindingInfo();
logStartup(host, port);

const app = new Elysia({
	serve: {
		hostname: host
	}
})
	.onError(({ code, status, error }) => {
		if (code === "NOT_FOUND")
			return status(404, {
				message: "The requested resource was not found."
			});
		if (code === "VALIDATION") return error.detail(error.message);
		return error;
	})
	.use(onAfterHandler)
	.use(cors())
	.use(
		openapi({
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
		})
	)
	.use(rootHandler)
	.use(pingHandler)
	.use(authHandler)
	.use(getVideoMetadataHandler)
	.use(songInfoHandler)
	.use(closeMileStoneHandler)
	.use(searchHandler)
	.use(getVideoSnapshotsHandler)
	.use(addSongHandler)
	.use(deleteSongHandler)
	.use(songEtaHandler)
	.get(
		"/song/:id",
		({ redirect, params }) => {
			console.log(`/song/${params.id}/info`);
			return redirect(`/song/${params.id}/info`, 302);
		},
		{
			detail: {
				hide: true
			}
		}
	)
	.get(
		"/video/:id",
		({ redirect, params }) => {
			return redirect(`/video/${params.id}/info`, 302);
		},
		{
			detail: {
				hide: true
			}
		}
	)
	.listen(15412);

export const VERSION = pkg.version;

export type App = typeof app;
