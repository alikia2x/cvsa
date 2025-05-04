import postgres from "postgres";
import { postgresConfigNpm } from "./pgConfigNew";

export const sql = postgres(postgresConfigNpm);

export const sqlTest = postgres(postgresConfigNpm);