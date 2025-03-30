import { Pool } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { postgresConfig } from "@core/db/pgConfig.ts";

const pool = new Pool(postgresConfig, 12);

export const db = pool;
