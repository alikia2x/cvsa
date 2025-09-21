import { Elysia } from "elysia";
import { getBindingInfo, logStartup } from "./startMessage";
import { pingHandler } from "@elysia/routes/ping";
import openapi from "@elysiajs/openapi";
import { cors } from "@elysiajs/cors";
import { getSongInfoHandler } from "@elysia/routes/song/info";

const [host, port] = getBindingInfo();
logStartup(host, port);

const app = new Elysia({
	serve: {
		hostname: host
	}
})
	.use(cors())
	.use(openapi())
	.use(pingHandler)
	.use(getSongInfoHandler)
	.listen(15412);

export const VERESION = "0.7.0";

export type App = typeof app;
