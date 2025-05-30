import { Hono } from "hono";
import type { TimingVariables } from "hono/timing";
import { startServer } from "./startServer.ts";
import { configureRoutes } from "./routing.ts";
import { configureMiddleWares } from "middleware";
import { notFoundRoute } from "routes/404.ts";

type Variables = TimingVariables;
const app = new Hono<{ Variables: Variables }>();

app.notFound(notFoundRoute);

configureMiddleWares(app);
configureRoutes(app);

await startServer(app);

export const VERSION = "0.4.6";
