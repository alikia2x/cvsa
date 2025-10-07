import { Elysia } from "elysia";
import { getBindingInfo, logStartup } from "./startMessage";
import { pingHandler } from "@elysia/routes/ping";
import openapi from "@elysiajs/openapi";
import { cors } from "@elysiajs/cors";
import { getSongInfoHandler } from "@elysia/routes/song/info";
import { rootHandler } from "@elysia/routes/root";
import { getVideoMetadataHandler } from "@elysia/routes/video/metadata";
import { closeMileStoneHandler } from "@elysia/routes/song/milestone";
import { serverTiming } from '@elysiajs/server-timing'

const [host, port] = getBindingInfo();
logStartup(host, port);

const app = new Elysia({
	serve: {
		hostname: host
	}
})
.use(serverTiming())
	.use(cors())
	.use(openapi())
	.use(rootHandler)
	.use(pingHandler)
	.use(getVideoMetadataHandler)
	.use(getSongInfoHandler)
	.use(closeMileStoneHandler)
	.listen(15412);

export const VERSION = "0.7.0";

export type App = typeof app;
