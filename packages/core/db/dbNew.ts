import postgres from "postgres";
import { postgresConfigCred, postgresConfig } from "./pgConfigNew";

export const sql = postgres(postgresConfig);

export const sqlCred = postgres(postgresConfigCred);

export const sqlTest = postgres(postgresConfig);
