import { authHandler } from "@backend/routes/auth";
import { pingHandler } from "@backend/routes/ping";
import { rootHandler } from "@backend/routes/root";
import { searchHandler } from "@backend/routes/search";
import { addSongHandler } from "@backend/routes/song/add";
import { deleteSongHandler } from "@backend/routes/song/delete";
import { songHandler } from "@backend/routes/song/info";
import { closeMileStoneHandler } from "@backend/routes/song/milestone";
import { songEtaHandler } from "@backend/routes/video/eta";
import { getVideoMetadataHandler } from "@backend/routes/video/metadata";
import { getVideoSnapshotsHandler } from "@backend/routes/video/snapshots";
import { cors } from "@elysiajs/cors";
import { Elysia, type ErrorHandler } from "elysia";
import { onAfterHandler } from "./onAfterHandle";
import { getBindingInfo, logStartup } from "./startMessage";
import "./mq";
import { openAPIMiddleware } from "@backend/middlewares/openapi";
import { getUnlabelledVideos, postVideoLabel } from "@backend/routes/video/label";
import pkg from "../package.json";

const [host, port] = getBindingInfo();
logStartup(host, port);

const errorHandler: ErrorHandler = ({ code, status, error }) => {
	if (code === "NOT_FOUND")
		return status(404, {
			message: "The requested resource was not found.",
		});
	if (code === "VALIDATION") return error.detail(error.message);
	return error;
};

const app = new Elysia({
	serve: {
		hostname: host,
	},
})
	.onError(errorHandler)
	.use(onAfterHandler)
	.use(cors())
	.use(openAPIMiddleware)
	.use(rootHandler)
	.use(pingHandler)
	.use(authHandler)
	.use(getVideoMetadataHandler)
	.use(songHandler)
	.use(closeMileStoneHandler)
	.use(searchHandler)
	.use(getVideoSnapshotsHandler)
	.use(addSongHandler)
	.use(deleteSongHandler)
	.use(songEtaHandler)
	.use(getUnlabelledVideos)
	.use(postVideoLabel)
	.get(
		"/song/:id",
		({ redirect, params }) => {
			console.log(`/song/${params.id}/info`);
			return redirect(`/song/${params.id}/info`, 302);
		},
		{
			detail: {
				hide: true,
			},
		}
	)
	.get(
		"/video/:id",
		({ redirect, params }) => {
			return redirect(`/video/${params.id}/info`, 302);
		},
		{
			detail: {
				hide: true,
			},
		}
	)
	.listen(15412);

export const VERSION = pkg.version;

export type App = typeof app;
