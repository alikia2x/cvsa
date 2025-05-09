import postgres from "postgres";
import { postgresConfigNpm, postgresCredConfigNpm } from "./config";

export const sql = postgres(postgresConfigNpm);
export const sqlCred = postgres(postgresCredConfigNpm);
