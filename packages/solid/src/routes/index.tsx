import { Layout } from "~/components/shell/Layout";
import { dbMain } from "~/drizzle";
import { bilibiliMetadata, latestVideoSnapshot } from "~db/main/schema";
import { and, desc, eq, gte, lt } from "drizzle-orm";
import { createAsync, query } from "@solidjs/router";
import { For, Suspense } from "solid-js";

const getVideoCloseTo1M = query(async () => {
	"use server";
	return dbMain
		.select()
		.from(bilibiliMetadata)
		.leftJoin(latestVideoSnapshot, eq(latestVideoSnapshot.aid, bilibiliMetadata.aid))
		.where(and(gte(latestVideoSnapshot.views, 900000), lt(latestVideoSnapshot.views, 1000000)))
		.orderBy(desc(latestVideoSnapshot.views))
		.limit(20);
}, "data");

export default function Home() {
	const videos = createAsync(() => getVideoCloseTo1M());
	return (
		<Layout>
			<title>中V档案馆</title>
			<main>
				<h1 class="text-5xl mb-8">中 V 档案馆</h1>
				<h2 class="text-3xl font-normal">传说助攻</h2>

				<div>
					<Suspense fallback={<div>Loading...</div>}>
						<For each={videos()}>{(video) => <li>{video.bilibili_metadata.aid}</li>}</For>
					</Suspense>
				</div>
			</main>
		</Layout>
	);
}
