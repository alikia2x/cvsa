"use server";

import { sql } from "@core/db/dbNew";
import { drizzle } from "drizzle-orm/postgres-js";

export const db = drizzle(sql);
export * from "./main/schema";
export * from "./type";
