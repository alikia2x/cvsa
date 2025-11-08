"use server";

import { drizzle } from "drizzle-orm/postgres-js";
import { sqlCred, sql } from "@core/db/dbNew";

export const dbMain = drizzle(sql);
export const dbCred = drizzle(sqlCred);
export const db = drizzle(sql);