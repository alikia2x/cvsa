"use server";

import { drizzle } from 'drizzle-orm/postgres-js';
import { sqlCred, sql } from "@cvsa/core";

export const dbMain = drizzle(sql);
export const dbCred = drizzle(sqlCred);