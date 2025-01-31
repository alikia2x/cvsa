// Import necessary modules
import { Client } from "https://deno.land/x/postgres/mod.ts";
import { Database } from "jsr:@db/sqlite@0.12";

const model_predicted_text = await Deno.readTextFile("./data/filter/model_predicted.jsonl");
const labels = model_predicted_text
	.split("\n")
	.map((line) => JSON.parse(line))
	.reduce((acc, item, _index) => {
		acc[item.aid] = item.label + 100;
		return acc;
	}, {} as { [key: number]: string });

interface SQLiteEntry {
	aid: number;
	bvid: number;
	status: "success" | "fail" | "error";
	data: string;
	timestamp: number;
}

// Define the SQLite database path
const sqliteDbPath = "./data/main.db";

// Define PostgreSQL connection details
const postgresConfig = {
	hostname: "localhost",
	database: "cvsa",
	user: "alikia",
	password: "",
	port: 5432,
};

// Function to connect to PostgreSQL
async function connectToPostgres() {
	const client = new Client(postgresConfig);
	await client.connect();
	return client;
}

const PER_PAGE = 1000;

/*
 * Function to format timestamp to PostgreSQL datetime format
 */
function formatDatetime(timestamp: number) {
	const date = new Date(timestamp * 1000);
	return date.toISOString().slice(0, 19).replace("T", " ");
}

// Function to read data from SQLite
function readFromSQLite(db: Database, page: number): Array<SQLiteEntry> {
	const offset = page * PER_PAGE;
	const query = `SELECT * FROM bili_info_crawl WHERE status = 'success' LIMIT ${PER_PAGE} OFFSET ${offset}`;
	const rows: SQLiteEntry[] = db.prepare(query).all();
	return rows;
}

// Function to insert data into PostgreSQL
async function insertIntoPostgres(client: Client, data: Array<SQLiteEntry>) {
	for (const entry of data) {
		try {
			const aid = entry.aid;
			const label = labels[aid];
			if (!label || label == 100) {
				//console.warn(`Skipped for aid ${entry.aid}.`)
				continue;
			}
			const jsonData = JSON.parse(entry.data);
			const bvid = entry.bvid;
			const views = jsonData.View.stat.view;
			const length = jsonData.View.pages[0].duration;
			const published_at = jsonData.View.pubdate;
			const query = `INSERT INTO songs (aid, bvid, views, length, published_at, type) VALUES ($1, $2, $3, $4, $5, $6)`;
			await client.queryObject(query, [aid, bvid, views, length, formatDatetime(published_at), label]);
			//console.log(`Inserted data for aid ${entry.aid}`)
		}
		catch (e) {
			console.error(`Error inserting data for aid ${entry.aid}:`, e)
		}
	}
}

// Main function to execute the script
async function main() {
  	// Connect to SQLite
	const sqliteDb = new Database(sqliteDbPath);

	// Connect to PostgreSQL
	const postgresClient = await connectToPostgres();

  	// Read data from SQLite
  	let page = 0;
	let data = [];

	do {
	    data = readFromSQLite(sqliteDb, page);
	    if (data.length > 0) {
			await insertIntoPostgres(postgresClient, data);
	        page++;
	    }
	} while (data.length > 0);

  	// Close PostgreSQL connection
	await postgresClient.end();

  	// Close SQLite connection
	sqliteDb.close();
}

// Run the main function
main().catch(console.error);