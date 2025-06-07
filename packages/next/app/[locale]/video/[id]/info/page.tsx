import { Suspense } from "react";
import Link from "next/link";
import { format } from "date-fns";
import { notFound } from "next/navigation";
import { Metadata } from "next";
import type { VideoInfoData } from "@cvsa/core";

const StatRow = ({ title, description }: { title: string; description?: number }) => {
	return (
		<div className="flex justify-between w-36">
			<span>{title}</span>
			<span>{description?.toLocaleString() ?? "N/A"}</span>
		</div>
	);
};

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
	const backendURL = process.env.BACKEND_URL;
	const { id } = await params;
	const res = await fetch(`${backendURL}/video/${id}/info`);
	if (!res.ok) {
		return {
			title: "页面未找到 - 中 V 档案馆"
		};
	}
	const data = await res.json();
	return {
		title: `${data.title} - 视频信息 - 中 V 档案馆`
	};
}

const VideoInfo = async ({ id }: { id: string }) => {
	const backendURL = process.env.BACKEND_URL;

	const res = await fetch(`${backendURL}/video/${id}/info`);

	if (!res.ok) {
		return notFound();
	}

	const data: VideoInfoData = await res.json();

	return (
		<div className="w-full lg:max-w-4xl lg:mx-auto lg:p-6 px-4">
			<h2 className="text-lg md:text-2xl mb-2">
				<Link href={`https://www.bilibili.com/video/${data.bvid}`}>{data.title || data.bvid}</Link>
			</h2>

			<p className="text-sm md:text-base font-normal text-on-surface-variant dark:text-dark-on-surface-variant mb-4">
				<span>
					{data.bvid} · av{data.aid}
				</span>
				<br />
				<span>发布于 {format(new Date(data.pubdate * 1000), "yyyy-MM-dd HH:mm:ss")}</span>
				<br />
				<span>播放：{(data.stat?.view ?? 0).toLocaleString()}</span> ·{" "}
				<span>弹幕：{(data.stat?.danmaku ?? 0).toLocaleString()}</span>
				<br />
				<span>
					分区: {data.tname}, tid{data.tid} · v2: {data.tname_v2}, tid
					{data.tid_v2}
				</span>
			</p>

			<img src={data.pic} referrerPolicy="no-referrer" className="rounded-lg" alt="Video cover" />

			<h3 className="font-medium text-lg mt-6 mb-1">简介</h3>
			<pre className="max-w-full wrap-anywhere break-all text-on-surface-variant text-sm md:text-base whitespace-pre-wrap dark:text-dark-on-surface-variant font-zh">
				{data.desc}
			</pre>

			<div className="mb-6 mt-4">
				<h2 className="mb-2 text-xl font-medium">统计数据</h2>
				<div className="flex flex-col gap-1">
					<StatRow title="播放" description={data.stat?.view} />
					<StatRow title="点赞" description={data.stat?.like} />
					<StatRow title="收藏" description={data.stat?.favorite} />
					<StatRow title="硬币" description={data.stat?.coin} />
					<StatRow title="评论" description={data.stat?.reply} />
					<StatRow title="弹幕" description={data.stat?.danmaku} />
					<StatRow title="分享" description={data.stat?.share} />
				</div>
			</div>
		</div>
	);
};

export default async function VideoPage({ params }: { params: Promise<{ id: string }> }) {
	const { id } = await params;
	return (
		<main className="flex flex-col items-center flex-grow gap-8 mt-10 md:mt-6 relative z-0 overflow-x-auto pb-8">
			<Suspense
				fallback={
					<main className="flex flex-col flex-grow items-center justify-center gap-8">
						<h1 className="text-4xl font-extralight">正在努力加载中……</h1>
					</main>
				}
			>
				<VideoInfo id={id} />
			</Suspense>
		</main>
	);
}
