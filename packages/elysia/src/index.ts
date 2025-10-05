import { Elysia } from "elysia";
import { getBindingInfo, logStartup } from "./startMessage";
import { pingHandler } from "@elysia/routes/ping";
import openapi from "@elysiajs/openapi";
import { cors } from "@elysiajs/cors";
import { getSongInfoHandler } from "@elysia/routes/song/info";
import { rootHandler } from "@elysia/routes/root";
import { getVideoMetadataHandler } from "@elysia/routes/video/metadata";

const [host, port] = getBindingInfo();
logStartup(host, port);

const app = new Elysia({
	serve: {
		hostname: host
	}
})
	.use(cors())
	.use(openapi())
	.use(rootHandler)
	.use(pingHandler)
	.use(getVideoMetadataHandler)
	.use(getSongInfoHandler)
	.listen(15412);

export const VERSION = "0.7.0";

export type App = typeof app;
