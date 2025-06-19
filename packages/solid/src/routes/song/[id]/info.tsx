import { getAllSnapshots } from "~/lib/db/snapshots/getAllSnapshots";
import { getAidFromBV } from "~/lib/db/bilibili_metadata/getAidFromBV";
import { getVideoMetadata } from "~/lib/db/bilibili_metadata/getVideoMetadata";
import { aidExists as idExists } from "~/lib/db/bilibili_metadata/aidExists";
import { BiliVideoMetadataType, VideoSnapshotType } from "@cvsa/core";
import { DateTime } from "luxon";
import { useParams } from "@solidjs/router";
import { createResource } from "solid-js";
import { Title } from "@solidjs/meta";
import { Suspense } from "solid-js";
import { For } from "solid-js";

const MetadataRow = ({ title, desc }: { title: string; desc: string | number | undefined | null }) => {
	if (!desc) return <></>;
	return (
		<tr>
			<td class="max-w-14 min-w-14 md:max-w-24 md:min-w-24 border dark:border-zinc-500 px-2 md:px-3 py-2 font-semibold">
				{title}
			</td>
			<td class="break-all max-w-[calc(100vw-4.5rem)] border dark:border-zinc-500 px-4 py-2">{desc}</td>
		</tr>
	);
};

function sleep(ms: number) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function VideoInfoPage() {
	const params = useParams();
	const { id } = params;
	const [data] = createResource(async () => {
		let videoInfo: BiliVideoMetadataType | null = null;
		let snapshots: VideoSnapshotType[] = [];
		await sleep(1000);

		async function getVideoAid(videoId: string | string[] | undefined) {
			if (!videoId) return null;
			const videoIdStr = Array.isArray(videoId) ? videoId[0] : videoId;
			if (videoIdStr?.startsWith("av")) {
				return parseInt(videoIdStr.slice(2));
			} else if (videoIdStr?.startsWith("BV")) {
				return getAidFromBV(videoIdStr);
			}
			return parseInt(videoIdStr);
		}

		const aid = await getVideoAid(id);

		if (!aid) {
			return null;
		}

		const exists = await idExists(aid);

		if (!exists) {
			return null;
		}

		try {
			const videoData = await getVideoMetadata(aid);
			const snapshotsData = await getAllSnapshots(aid);
			videoInfo = videoData;
			if (snapshotsData) {
				snapshots = snapshotsData;
			}
		} catch (e) {
			console.error(e);
		}

		if (!videoInfo) {
			return null;
		}
		let title = "";
		const backendURL = process.env.BACKEND_URL;
		const res = await fetch(`${backendURL}/video/${id}/info`);
		if (!res.ok) {
			title = "页面未找到 - 中 V 档案馆";
		}
		const data = await res.json();

		title = `${data.title} - 歌曲信息 - 中 V 档案馆`;

		return {
			v: videoInfo,
			s: snapshots,
			t: title
		};
	});
	return (
		<main class="flex flex-col items-center min-h-screen gap-8 mt-10 md:mt-6 relative z-0 overflow-x-auto pb-8">
			<div>hi</div>
			<div class="w-full lg:max-w-4xl lg:mx-auto lg:p-6">
				<Suspense fallback={<div>loading</div>}>
					<Title>{data()?.t}</Title>
					<h1 class="text-2xl font-medium ml-2 mb-4">
						视频信息:{" "}
						<a href={`https://www.bilibili.com/video/av${data()?.v.aid}`} class="underline">
							av{data()?.v.aid}
						</a>
					</h1>

					<div class="mb-6">
						<h2 class="px-2 mb-2 text-xl font-medium">基本信息</h2>
						<div class="overflow-x-auto max-w-full px-2">
							<table class="table-fixed">
								<tbody>
									<MetadataRow title="ID" desc={data()?.v.id} />
									<MetadataRow title="av 号" desc={data()?.v.aid} />
									<MetadataRow title="BV 号" desc={data()?.v.bvid} />
									<MetadataRow title="标题" desc={data()?.v.title} />
									<MetadataRow title="描述" desc={data()?.v.description} />
									<MetadataRow title="UID" desc={data()?.v.uid} />
									<MetadataRow title="标签" desc={data()?.v.tags} />
									<MetadataRow
										title="发布时间"
										desc={
											data()?.v.published_at
												? DateTime.fromJSDate(new Date(data()?.v.published_at || "")).toFormat(
														"yyyy-MM-dd HH:mm:ss"
													)
												: null
										}
									/>
									<MetadataRow title="时长 (秒)" desc={data()?.v.duration} />
									<MetadataRow
										title="创建时间"
										desc={DateTime.fromJSDate(new Date(data()?.v.created_at || "")).toFormat(
											"yyyy-MM-dd HH:mm:ss"
										)}
									/>
									<MetadataRow title="封面" desc={data()?.v?.cover_url} />
								</tbody>
							</table>
						</div>
					</div>

					<div>
						<h2 class="px-2 mb-2 text-xl font-medium">播放量历史数据</h2>

						<div class="overflow-x-auto px-2">
							<table class="table-auto w-full">
								<thead>
									<tr>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">创建时间</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">观看</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">硬币</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">点赞</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">收藏</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">分享</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">弹幕</th>
										<th class="border dark:border-zinc-500 px-4 py-2 font-medium">评论</th>
									</tr>
								</thead>
								<tbody>
									<For each={data()?.s}>
										{(snapshot) => (
											<tr>
												<td class="border dark:border-zinc-500 px-4 py-2">
													{DateTime.fromJSDate(snapshot.created_at).toFormat(
														"yyyy-MM-dd HH:mm:ss"
													)}
												</td>
												<td class="border dark:border-zinc-500 px-4 py-2">{snapshot.views}</td>
												<td class="border dark:border-zinc-500 px-4 py-2">{snapshot.coins}</td>
												<td class="border dark:border-zinc-500 px-4 py-2">{snapshot.likes}</td>
												<td class="border dark:border-zinc-500 px-4 py-2">
													{snapshot.favorites}
												</td>
												<td class="border dark:border-zinc-500 px-4 py-2">{snapshot.shares}</td>
												<td class="border dark:border-zinc-500 px-4 py-2">
													{snapshot.danmakus}
												</td>
												<td class="border dark:border-zinc-500 px-4 py-2">
													{snapshot.replies}
												</td>
											</tr>
										)}
									</For>
								</tbody>
							</table>
						</div>
					</div>
				</Suspense>
			</div>
		</main>
	);
}
