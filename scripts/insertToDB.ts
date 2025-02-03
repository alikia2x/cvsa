// Import necessary modules
import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { Database } from "jsr:@db/sqlite@0.12";

interface SQLiteEntry {
    aid: number;
    bvid: string;
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

function aidExistsInPG(allAidsSet: Set<number>, aid: number) {
    return allAidsSet.has(aid);
}

async function getAllAidsFromPG(client: Client) {
    const query = `SELECT aid FROM songs`;
    const result = await client.queryArray(query);
    const rows = result.rows as Array<number[]>;
    return rows.map(item => Number(item[0]));
}

async function insertToAllData(client: Client, aid: number, bvid: string, desc: string, uid: number, tags: string, title: string) {
    const query = `INSERT INTO all_data (aid, bvid, description, uid, tags, title) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (aid) DO NOTHING;`;
    await client.queryObject(query, [aid, bvid, desc, uid, tags, title]);
}

async function insertLabellingResult(client: Client, aid: number, label: number) {
    const query = `INSERT INTO labelling_result (aid, label, model_version) VALUES ($1, $2, 'v3.9') ON CONFLICT (aid, model_version) DO NOTHING;`;
    await client.queryObject(query, [aid, label]);
}

// Function to insert data into PostgreSQL
async function insertIntoPostgres(client: Client, data: Array<SQLiteEntry>, labels: { [key: number]: number | undefined }, allAidsSet: Set<number>) {
    for (const entry of data) {
        try {
            const aid = entry.aid;
            const modelLabel = labels[aid] ?? null;
            const jsonData = JSON.parse(entry.data);
            const bvid = entry.bvid;
            const views = jsonData.View.stat.view;
            const length = jsonData.View.pages[0].duration;
            const published_at = jsonData.View.pubdate;
            const uid = jsonData.Card.card.mid;
            const tags: string = (jsonData.Tags as Array<{ tag_type: string; tag_name: string }>)
                .filter(tag => ["old_channel", "topic"].includes(tag.tag_type))
                .map(tag => tag.tag_name)
                .join(",");
            const title = jsonData.View.title;
            const desc = jsonData.View.desc;
            await insertToAllData(client, aid, bvid, desc, uid, tags, title);
            if (modelLabel !== null) {
                await insertLabellingResult(client, aid, modelLabel - 100);
            }
            const aidExists = aidExistsInPG(allAidsSet, aid);
            if (!aidExists && modelLabel !== 100) {
                const query = `INSERT INTO songs (aid, bvid, views, length, published_at, type) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (aid) DO NOTHING;`;
                await client.queryObject(query, [aid, bvid, views, length, formatDatetime(published_at), modelLabel]);
            }
            //console.log(`Inserted data for aid ${entry.aid}`)
        } catch (e) {
            console.error(`Error inserting data for aid ${entry.aid}:`, e);
        }
    }
}

// Main function to execute the script
async function main() {
    const model_predicted_text = await Deno.readTextFile("./data/filter/model_predicted.jsonl");
    const labels = model_predicted_text
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line))
        .reduce((acc, item, _index) => {
            acc[item.aid] = item.label + 100;
            return acc;
        }, {} as { [key: number]: number | undefined });

    // Connect to SQLite
    const sqliteDb = new Database(sqliteDbPath, { int64: true });

    // Connect to PostgreSQL
    const postgresClient = await connectToPostgres();
    const allAids = await getAllAidsFromPG(postgresClient);
    const allAidsSet = new Set(allAids);

    // Read data from SQLite
    let page = 0;
    let data = [];

    do {
        data = readFromSQLite(sqliteDb, page);
        if (data.length > 0) {
            await insertIntoPostgres(postgresClient, data, labels, allAidsSet);
            page++;
        }
    } while (data.length > 0);

    // Close PostgreSQL connection
    await postgresClient.end();

    // Close SQLite connection
    sqliteDb.close();
}

// Run the main function
main().catch(console.error)