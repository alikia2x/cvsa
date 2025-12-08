import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
	index("home/home.tsx"),
	route("project/:id", "projects/projectPage.tsx"),
	route("project/new", "projects/newProject.tsx"),
	route("login", "login/page.tsx"),
	route("admin/users", "admin/users.tsx"),
	route("setup", "setup/setup.tsx"),
	route("profile", "user/profile.tsx"),
	route("logout", "logout/logout.tsx"),
	route("project/:id/settings", "projects/settings.tsx")
] satisfies RouteConfig;
