import postgres from "postgres";
import { postgresConfigNpm } from "./config";

const sql = postgres(postgresConfigNpm);

export default sql;
