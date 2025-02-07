import { Client } from "https://deno.land/x/postgres@v0.19.3/mod.ts";
import { getLatestVideos } from "lib/net/getLatestVideos.ts";
import { insertIntoAllData, videoExistsInAllData } from "lib/db/allData.ts";
import { sleep } from "lib/utils/sleep.ts";

const requiredEnvVars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"];

const unsetVars = requiredEnvVars.filter((key) => Deno.env.get(key) === undefined);

if (unsetVars.length > 0) {
	throw new Error(`Missing required environment variables: ${unsetVars.join(", ")}`);
}

const databaseHost = Deno.env.get("DB_HOST")!;
const databaseName = Deno.env.get("DB_NAME")!;
const databaseUser = Deno.env.get("DB_USER")!;
const databasePassword = Deno.env.get("DB_PASSWORD")!;
const databasePort = Deno.env.get("DB_PORT")!;

const postgresConfig = {
	hostname: databaseHost,
	port: parseInt(databasePort),
	database: databaseName,
	user: databaseUser,
	password: databasePassword,
};

async function connectToPostgres() {
	const client = new Client(postgresConfig);
	await client.connect();
	return client;
}

export async function insertLatestVideos() {
	const client = await connectToPostgres();
	let page = 334;
	let failCount = 0;
	while (true) {
		try {
			const videos = await getLatestVideos(page, 10);
			if (videos == null) {
				failCount++;
				if (failCount > 5) {
					break;
				}
				continue;
			}
			if (videos.length == 0) {
				console.warn("No more videos found");
				break;
			}
			let allExists = true;
			for (const video of videos) {
				const videoExists = await videoExistsInAllData(client, video.aid);
				if (!videoExists) {
					allExists = false;
					insertIntoAllData(client, video);
				}
			}
			if (allExists) {
				console.log("All videos already exist in all_data, stop crawling.");
				break;
			}
			console.log(`Page ${page} crawled, total: ${(page - 1) * 20 + videos.length} videos.`);
			page++;
		} catch (error) {
			console.error(error);
			failCount++;
			if (failCount > 5) {
				break;
			}
			continue;
		}
		finally {
			await sleep(Math.random() * 4000 + 1000);
		}
	}
}


insertLatestVideos();