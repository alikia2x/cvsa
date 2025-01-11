import path from "node:path";
import { Database } from "jsr:@db/sqlite@0.12";
import { getBiliBiliVideoInfo } from "../../crawler/bilibili/videoInfo.ts";

const aidPath = path.join("./data/2025010104_c30_aids.txt");

const db = new Database("./data/main.db");

async function insertAidsToDB() {
    const aidRawcontent = await Deno.readTextFile(aidPath);
    const aids = aidRawcontent
        .split("\n")
        .filter((line) => line.length > 0)
        .map((line) => parseInt(line));

    // Insert aids into database with status 'pending'
    // if bili_info_crawl contains at least one row, skip it
    if (db.prepare("SELECT * FROM bili_info_crawl").all().length == 0) {
        const insertStmt = db.prepare(`
            INSERT OR IGNORE INTO bili_info_crawl (aid, status)
            VALUES (?, 'pending')
          `);
        for (const aid of aids) {
            insertStmt.run(aid);
        }
    }

    const aidsInDB = db
        .prepare("SELECT aid FROM bili_info_crawl WHERE status = 'pending' OR status = 'failed'")
        .all()
        .map((row) => row.aid) as number[];

    const totalAids = aidsInDB.length;
    let processedAids = 0;
    const startTime = Date.now();

    // Update database with video info
    for (const aid of aidsInDB) {
        try {
            const res = await getBiliBiliVideoInfo(aid);
            if (res?.data.code !== 0) {
                const data = res?.data;
                db.prepare(
                    `
                        UPDATE bili_info_crawl
                        SET status = 'error',
                        data = ?
                        WHERE aid = ?
                    `
                ).run(aid, JSON.stringify(data));
            } else {
                const data = res.data.data;
                db.prepare(
                    `
                        UPDATE bili_info_crawl
                        SET status = 'success',
                        bvid = ?,
                        data = ?
                        WHERE aid = ?
                    `
                ).run(data.View.bvid, JSON.stringify(data), aid);
            }
        } catch (error) {
            console.error(`Error updating aid ${aid}: ${error}`);
            try {
                db.prepare(
                    `
                        UPDATE bili_info_crawl
                        SET status = 'failed'
                        WHERE aid = ?
                    `
                ).run(aid);
            }
            catch (error) {
                console.error(`Error wrting to db for aid ${aid}: ${error}`);
            }
        } finally {
            processedAids++;
            const elapsedTime = Date.now() - startTime;
            const elapsedSeconds = Math.floor(elapsedTime / 1000);
            const elapsedMinutes = Math.floor(elapsedSeconds / 60);
            const elapsedHours = Math.floor(elapsedMinutes / 60);
            const remainingAids = totalAids - processedAids;

            // Calculate ETA
            const averageTimePerAid = elapsedTime / processedAids;
            const eta = remainingAids * averageTimePerAid;
            const etaSeconds = Math.floor(eta / 1000);
            const etaMinutes = Math.floor(etaSeconds / 60);
            const etaHours = Math.floor(etaMinutes / 60);

            // Output progress
            const progress = `${processedAids}/${totalAids}, ${(processedAids / totalAids * 100).toFixed(2)}%, elapsed ${elapsedHours.toString().padStart(2, '0')}:${(elapsedMinutes % 60).toString().padStart(2, '0')}:${(elapsedSeconds % 60).toString().padStart(2, '0')}, ETA ${etaHours}h${(etaMinutes % 60).toString().padStart(2, '0')}m`;

            if (Math.random() > 0.95) {
                console.log("Sleeping...");
                const time = Math.random() * 5 * 1000;
                await new Promise((resolve) => setTimeout(resolve, time));
            }
            console.log(`Updated aid ${aid}, ${progress}`);
        }
    }
}

insertAidsToDB();
