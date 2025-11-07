import Elysia from "elysia";
import { loginHandler } from "./login";
import { logoutHandler } from "./logout";

export const authHandler = new Elysia().use(loginHandler).use(logoutHandler);
