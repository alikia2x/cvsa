import postgres from "postgres";
import { postgresConfig } from "./pgConfigNew";

export const sql = postgres(postgresConfig);