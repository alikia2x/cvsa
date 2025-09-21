import { Layout } from "~/components/layout";
import { dbMain } from "~db/index";
import { bilibiliMetadata, latestVideoSnapshot } from "~db/main/schema";
import { and, desc, eq, gte, lt } from "drizzle-orm";
import { A, createAsync, query, RouteDefinition } from "@solidjs/router";
import { Component, For, Suspense } from "solid-js";
import { BilibiliMetadataType, LatestVideoSnapshotType } from "~db/outerSchema";
import { Card, CardContent, CardMedia, Typography } from "@m3-components/solid";

const getVideoCloseTo1M = query(async () => {
	"use server";
	return dbMain
		.select()
		.from(bilibiliMetadata)
		.innerJoin(latestVideoSnapshot, eq(latestVideoSnapshot.aid, bilibiliMetadata.aid))
		.where(and(gte(latestVideoSnapshot.views, 900000), lt(latestVideoSnapshot.views, 1000000)))
		.orderBy(desc(latestVideoSnapshot.views))
		.limit(20);
}, "videosCloseTo1M");

interface VideoCardProps {
	video: {
		bilibili_metadata: BilibiliMetadataType;
		latest_video_snapshot: LatestVideoSnapshotType;
	};
}

export const route = {
	preload: () => getVideoCloseTo1M()
} satisfies RouteDefinition;

const VideoCard: Component<VideoCardProps> = (props) => {
	return (
		<Card variant="outlined" class="w-64 h-64 grow-0 shrink-0 basis-64">
			<CardMedia
				class="w-64 h-32 object-cover"
				round={false}
				src={props.video.bilibili_metadata.coverUrl || ""}
				referrerpolicy="no-referrer"
			/>
			<CardContent class="py-3 px-4">
				<A href={`/song/av${props.video.bilibili_metadata.aid}/info`}>
					<Typography.Body variant="large" class="text-wrap">
						{props.video.bilibili_metadata.title}
					</Typography.Body>
				</A>
				<span>{props.video.latest_video_snapshot.views} 播放</span>
			</CardContent>
		</Card>
	);
};

export default function Home() {
	const videos = createAsync(() => getVideoCloseTo1M());
	return (
		<Layout>
			<title>中V档案馆</title>
			<main class="w-full pt-20 lg:max-w-3xl xl:max-w-4xl 2xl:max-w-6xl lg:mx-auto">
				<h1 class="text-4xl mb-8">中 V 档案馆</h1>
				<h2 class="text-2xl font-normal">传说助攻</h2>
				<div
					class="flex overflow-x-auto overflow-y-hidden gap-4 whitespace-nowrap w-full
						py-2 px-4 mt-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
				>
					<Suspense fallback={<div>Loading...</div>}>
						<For each={videos()}>{(video) => <VideoCard video={video} />}</For>
					</Suspense>
				</div>
			</main>
		</Layout>
	);
}
