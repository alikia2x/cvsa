"use server";

import { drizzle } from "drizzle-orm/postgres-js";
import { sql } from "@core/db/dbNew";

export const db = drizzle(sql);
export * from "./main/schema";
export * from "./type";