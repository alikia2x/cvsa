import { Pool } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import {postgresConfig} from "lib/db/pgConfig.ts";

const pool = new Pool(postgresConfig, 10);

export const db = pool;
