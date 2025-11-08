import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
	index("routes/home.tsx"),
	route("song/:id/info", "routes/song/[id]/info.tsx"),
	route("chart-demo", "routes/chartDemo.tsx"),
	route("search", "routes/search/index.tsx"),
] satisfies RouteConfig;
