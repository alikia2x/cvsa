import { postgresConfig } from "@core/db/pgConfigNew";
import logger from "@core/log";
import { $, S3Client } from "bun";
import dayjs from "dayjs";
import { env } from "./env";

const AK = env.OSS_ACCESS_KEY_ID;
const SK = env.OSS_ACCESS_KEY_SECRET;
const ENDPOINT = env.BACKUP_S3_ENDPOINT;
const REGION = env.BACKUP_S3_REGION;
const BUCKET = env.BACKUP_S3_BUCKET;
const DIR = env.BACKUP_DIR;

const CONFIG = {
	localBackupDir: DIR,
	retentionDaily: 3,
	s3: {
		bucket: BUCKET,
		endpoint: ENDPOINT,
		region: REGION,
	},
};

const { username, password, host, port, database } = postgresConfig;
const dbUri = `postgresql://${username}:${encodeURIComponent(password)}@${host}:${port}/${database}`;

const s3 = new S3Client({
	accessKeyId: AK,
	bucket: CONFIG.s3.bucket,
	endpoint: CONFIG.s3.endpoint,
	region: CONFIG.s3.region,
	secretAccessKey: SK,
	virtualHostedStyle: true,
});

const getDayStr = (): string => {
	return dayjs().format("YYYY-MM-DD");
};

const getMonthStr = (): string => {
	return dayjs().format("YYYY-MM");
};

async function runBackup() {
	const dayStr = getDayStr();
	const monthStr = getMonthStr();
	const fileName = `cvsa_${dayStr}.dump`;
	const filePath = `${CONFIG.localBackupDir}/${fileName}`;
	const localDumpfile = Bun.file(filePath);

	logger.log(`Starting backup...`);

	if (!(await localDumpfile.exists())) {
		logger.log(`Creating dump ${localDumpfile.name}...`);
		const cmd = $`pg_dump -d ${dbUri} -Fc -n public > ${filePath}`;

		await cmd;
	}

	const monthlyBackupFile = s3.file(`dump/monthly/${monthStr}`);

	if (!(await monthlyBackupFile.exists())) {
		logger.log(`Uploading ${filePath} to ${monthlyBackupFile.name}`);
		await monthlyBackupFile.write(localDumpfile);
	}

	const dailyBackupFile = s3.file(`dump/daily/${dayStr}`);

	if (!(await dailyBackupFile.exists())) {
		logger.log(`Uploading ${filePath} to ${dailyBackupFile.name}`);
		await dailyBackupFile.write(localDumpfile);
	}
}

async function rotateS3Backups() {
	const dailyBackups = await s3.list({
		maxKeys: 1000,
		prefix: "dump/daily/",
	});
	if (!dailyBackups.contents) {
		logger.log("No daily backups found");
		return;
	}
	logger.log(`Found ${dailyBackups.contents.length} daily backups`);
	for (const content of dailyBackups.contents) {
		const key = content.key;
		if (!key) {
			continue;
		}
		const dateStr = key.split("/").at(-1);
		const date = dayjs(dateStr, "YYYY-MM-DD");
		if (date.isBefore(dayjs().subtract(CONFIG.retentionDaily, "day"))) {
			logger.log(`Deleting daily backup ${key}`);
			await s3.delete(key);
		}
	}

	if (dailyBackups.isTruncated) {
		logger.log("Still more daily backups");
		await rotateS3Backups();
	}
	logger.log("Daily backups rotated");
}

async function main() {
	try {
		await runBackup();
		await rotateS3Backups();
	} catch (err) {
		logger.error(err);
		process.exit(1);
	}
}

await main();
process.exit();
