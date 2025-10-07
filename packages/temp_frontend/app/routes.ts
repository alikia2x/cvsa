import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
	index("routes/home.tsx"),
	route("song/:id/info", "routes/song/[id]/info.tsx"),
	route("song/:id/data", "routes/song/[id]/data.tsx"),
	route("chart-demo", "routes/chartDemo.tsx"),
] satisfies RouteConfig;
