import { Elysia } from "elysia";
import { getBindingInfo, logStartup } from "./startMessage";
import { pingHandler } from "@elysia/routes/ping";
import openapi from "@elysiajs/openapi";
import { cors } from "@elysiajs/cors";
import { songInfoHandler } from "@elysia/routes/song/info";
import { rootHandler } from "@elysia/routes/root";
import { getVideoMetadataHandler } from "@elysia/routes/video/metadata";
import { closeMileStoneHandler } from "@elysia/routes/song/milestone";
import { authHandler } from "@elysia/routes/auth";
import { onAfterHandler } from "./onAfterHandle";
import { searchHandler } from "@elysia/routes/search";
import { getVideoSnapshotsHandler } from "@elysia/routes/video/snapshots";

const [host, port] = getBindingInfo();
logStartup(host, port);

const app = new Elysia({
	serve: {
		hostname: host
	}
})
	.use(onAfterHandler)
	.use(cors())
	.use(openapi())
	.use(rootHandler)
	.use(pingHandler)
	.use(authHandler)
	.use(getVideoMetadataHandler)
	.use(songInfoHandler)
	.use(closeMileStoneHandler)
	.use(searchHandler)
	.use(getVideoSnapshotsHandler)
	.listen(15412);

export const VERSION = "0.7.0";

export type App = typeof app;
