import Elysia from "elysia";
import { loginHandler } from "./login";
import { logoutHandler } from "./logout";
import { getCurrentUserHandler } from "./user";

export const authHandler = new Elysia()
	.use(loginHandler)
	.use(logoutHandler)
	.use(getCurrentUserHandler);
