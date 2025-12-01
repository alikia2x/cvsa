import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
	index("routes/home/index.tsx"),
	route("song/:id/info", "routes/song/[id]/info/index.tsx"),
	route("song/:id/add", "routes/song/[id]/add.tsx"),
	route("search", "routes/search/index.tsx"),
	route("login", "routes/login.tsx"),
	route("video/:id/info", "routes/video/[id]/info/index.tsx"),
	route("time-calculator", "routes/time-calculator.tsx"),
	route("labelling", "routes/labelling/index.tsx"),
] satisfies RouteConfig;
