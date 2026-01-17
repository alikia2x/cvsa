import { createEnv } from "@t3-oss/env-core";
import { z } from "zod";

export const env = createEnv({
	runtimeEnv: Bun.env,
	server: {
		BACKUP_DIR: z.string(),
		BACKUP_S3_BUCKET: z.string(),
		BACKUP_S3_ENDPOINT: z.string(),
		BACKUP_S3_REGION: z.string(),
		OSS_ACCESS_KEY_ID: z.string(),
        OSS_ACCESS_KEY_SECRET: z.string(),
		BACKUP_CONTAINER: z.string(),
	},
});
