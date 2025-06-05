import { expect, test } from "vitest";
import { sqlTest as sql } from "@core/db/dbNew";
import { SnapshotScheduleType } from "@core/db/schema";
import { bulkSetSnapshotStatus } from "db/snapshotSchedule";

const mockSnapshotSchedules: SnapshotScheduleType[] = [
	{
		id: 1,
		aid: 1234567890,
		type: "normal",
		created_at: "2025-05-04T00:00:00.000Z",
		started_at: "2025-05-04T06:00:00.000Z",
		finished_at: "2025-05-04T06:03:27.000Z",
		status: "completed"
	},
	{
		id: 2,
		aid: 9876543210,
		type: "archive",
		created_at: "2025-05-03T12:00:00.000Z",
		started_at: "2025-05-03T13:00:00.000Z",
		finished_at: null,
		status: "failed"
	},
	{
		id: 3,
		aid: 1122334455,
		type: "milestone",
		created_at: "2025-05-01T08:00:00.000Z",
		started_at: "2025-05-01T08:12:00.000Z",
		finished_at: null,
		status: "processing"
	}
];

const databasePreparationQuery = `
    CREATE SEQUENCE "snapshot_schedule_id_seq"
        START WITH 1
        INCREMENT BY 1
        MINVALUE 1
        MAXVALUE 9223372036854775807
        CACHE 1;

    CREATE TABLE "snapshot_schedule"(
        "id" bigint DEFAULT nextval('snapshot_schedule_id_seq'::regclass) NOT NULL,
        "aid" bigint NOT NULL,
        "type" text,
        "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
        "started_at" timestamp with time zone,
        "finished_at" timestamp with time zone,
        "status" text DEFAULT 'pending'::text NOT NULL
    );

    CREATE INDEX idx_snapshot_schedule_aid ON snapshot_schedule USING btree (aid);
    CREATE INDEX idx_snapshot_schedule_started_at ON snapshot_schedule USING btree (started_at);
    CREATE INDEX idx_snapshot_schedule_status ON snapshot_schedule USING btree (status);
    CREATE INDEX idx_snapshot_schedule_type ON snapshot_schedule USING btree (type);
    CREATE UNIQUE INDEX snapshot_schedule_pkey ON snapshot_schedule USING btree (id);
`;

const cleanUpQuery = `
    DROP SEQUENCE IF EXISTS "snapshot_schedule_id_seq" CASCADE;
    DROP TABLE IF EXISTS "snapshot_schedule" CASCADE;
`;

async function testMocking() {
	await sql.begin(async (tx) => {
		await tx.unsafe(cleanUpQuery).simple();
		await tx.unsafe(databasePreparationQuery).simple();

		await tx`
            INSERT INTO snapshot_schedule 
            ${sql(mockSnapshotSchedules, "aid", "created_at", "finished_at", "id", "started_at", "status", "type")}
        `;

		await tx`
            ROLLBACK;
        `;

		await tx.unsafe(cleanUpQuery).simple();
		return;
	});
}

async function testBulkSetSnapshotStatus() {
	return await sql.begin(async (tx) => {
		await tx.unsafe(cleanUpQuery).simple();
		await tx.unsafe(databasePreparationQuery).simple();

		await tx`
            INSERT INTO snapshot_schedule 
            ${sql(mockSnapshotSchedules, "aid", "created_at", "finished_at", "id", "started_at", "status", "type")}
        `;

		const ids = [1, 2, 3];

		await bulkSetSnapshotStatus(tx, ids, "pending");

		const rows = tx<{ status: string }[]>`
            SELECT status FROM snapshot_schedule WHERE id = 1;
        `.execute();

		await tx`
            ROLLBACK;
        `;

		await tx.unsafe(cleanUpQuery).simple();
		return rows;
	});
}

test("data mocking works", async () => {
	await testMocking();
	expect(() => {}).not.toThrowError();
});

test("bulkSetSnapshotStatus core logic works smoothly", async () => {
	const rows = await testBulkSetSnapshotStatus();
	expect(rows.every((item) => item.status === "pending")).toBe(true);
});
