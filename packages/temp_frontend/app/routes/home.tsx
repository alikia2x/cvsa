import { Layout } from "@/components/Layout";
import type { Route } from "./+types/home";
import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { treaty } from "@elysiajs/eden";
import type { App } from "@backend/src";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDateTime } from "@/components/SearchResults";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { addHoursToNow, formatHours } from "./song/[id]/info";

// @ts-ignore idk
const app = treaty<App>(import.meta.env.VITE_API_URL!);

type CloseMilestoneInfo = Awaited<ReturnType<ReturnType<(typeof app.songs)["close-milestone"]>["get"]>>["data"];
type CloseMilestoneError = Awaited<ReturnType<ReturnType<(typeof app.songs)["close-milestone"]>["get"]>>["error"];

export function meta({}: Route.MetaArgs) {
	return [{ title: "中V档案馆" }];
}

type MilestoneType = "dendou" | "densetsu" | "shinwa";

const milestoneConfig = {
	dendou: { name: "殿堂", range: [90000, 99999], target: 100000 },
	densetsu: { name: "传说", range: [900000, 999999], target: 1000000 },
	shinwa: { name: "神话", range: [5000000, 9999999], target: 10000000 },
};

export default function Home() {
	const [input, setInput] = useState("");
	const [milestoneType, setMilestoneType] = useState<MilestoneType>("shinwa");
	const [closeMilestoneInfo, setCloseMilestoneInfo] = useState<CloseMilestoneInfo>();
	const [closeMilestoneError, setCloseMilestoneError] = useState<CloseMilestoneError>();
	const [isLoading, setIsLoading] = useState(false);

	const fetchMilestoneData = async (type: MilestoneType) => {
		setIsLoading(true);
		setCloseMilestoneError(undefined);
		const { data, error } = await app.songs["close-milestone"]({ type }).get();
		if (error) {
			setCloseMilestoneError(error);
		} else {
			setCloseMilestoneInfo(data);
		}
		setIsLoading(false);
	};

	useEffect(() => {
		fetchMilestoneData(milestoneType);
	}, [milestoneType]);

	const MilestoneVideoCard = ({ video }: { video: NonNullable<CloseMilestoneInfo>[number] }) => {
		const config = milestoneConfig[milestoneType];
		const remainingViews = config.target - video.eta.currentViews;
		const progressPercentage = (video.eta.currentViews / config.target) * 100;

		return (
			<Card className="px-3 max-md:py-3 md:px-4 my-4 gap-0">
				<div className="w-full flex items-start space-x-4 mb-4">
					{video.bilibili_metadata.coverUrl && (
						<img
							src={video.bilibili_metadata.coverUrl}
							alt="视频封面"
							className="h-25 w-40 rounded-sm object-cover shrink-0"
							referrerPolicy="no-referrer"
							loading="lazy"
						/>
					)}
					<div className="flex flex-col w-full justify-between">
						<h3 className="text-sm sm:text-lg font-medium line-clamp-2 text-wrap mb-2">
							<a href={`/song/av${video.bilibili_metadata.aid}/info`} className="hover:underline">
								{video.bilibili_metadata.title}
							</a>
						</h3>

						<div className="space-y-2 text-xs text-muted-foreground">
							<div className="flex items-center justify-between">
								<span>当前播放: {video.eta.currentViews.toLocaleString()}</span>
								<span>目标: {config.target.toLocaleString()}</span>
							</div>

							<Progress value={progressPercentage} />
						</div>
					</div>
				</div>
				<div className="grid grid-cols-2 gap-5 text-xs text-muted-foreground mb-2">
					<div>
						<p>剩余播放: {remainingViews.toLocaleString()}</p>
						<p>预计达成: {formatHours(video.eta.eta)}</p>
					</div>
					<div>
						<p>播放速度: {Math.round(video.eta.speed)}/小时</p>
						<p>达成时间: {addHoursToNow(video.eta.eta)}</p>
					</div>
				</div>

				<div className="flex gap-4 text-xs text-muted-foreground">
					{video.bilibili_metadata.publishedAt && (
						<span className="stat-num">
							发布于 {formatDateTime(new Date(video.bilibili_metadata.publishedAt))}
						</span>
					)}
					<a
						href={`https://www.bilibili.com/video/av${video.bilibili_metadata.aid}`}
						target="_blank"
						rel="noopener noreferrer"
						className="text-pink-400 text-xs hover:underline"
					>
						观看视频
					</a>
					<a
						href={`/song/av${video.bilibili_metadata.aid}/info`}
						className="text-xs text-secondary-foreground hover:underline"
					>
						查看详情
					</a>
				</div>
			</Card>
		);
	};

	const MilestoneVideos = () => {
		if (isLoading) {
			return (
				<div className="space-y-4">
					{[1, 2, 3].map((i) => (
						<div
							key={i}
							className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm border border-gray-200 dark:border-neutral-700 p-4"
						>
							<div className="flex items-start space-x-4">
								<Skeleton className="h-21 w-36 rounded-sm" />
								<div className="flex-1 space-y-2">
									<Skeleton className="h-6 w-3/4" />
									<Skeleton className="h-4 w-full" />
									<Skeleton className="h-4 w-2/3" />
									<Skeleton className="h-4 w-1/2" />
								</div>
							</div>
						</div>
					))}
				</div>
			);
		}

		if (closeMilestoneError) {
			return (
				<div className="text-center py-8">
					<p className="text-red-500">加载失败: {closeMilestoneError.value?.message || "未知错误"}</p>
					<Button variant="outline" className="mt-4" onClick={() => fetchMilestoneData(milestoneType)}>
						重试
					</Button>
				</div>
			);
		}

		if (!closeMilestoneInfo || closeMilestoneInfo.length === 0) {
			return (
				<div className="text-center py-8">
					<p className="text-secondary-foreground">暂无接近{milestoneConfig[milestoneType].name}的视频</p>
				</div>
			);
		}

		return (
			<div className="space-y-4">
				<p className="text-xs text-muted-foreground">
					找到 {closeMilestoneInfo.length} 个接近{milestoneConfig[milestoneType].name}的视频
				</p>
				<ScrollArea className="h-140 w-full">
					{closeMilestoneInfo.map((video) => (
						<MilestoneVideoCard key={video.bilibili_metadata.aid} video={video} />
					))}
				</ScrollArea>
			</div>
		);
	};

	return (
		<Layout>
			<h2 className="text-2xl mt-5 mb-6">小工具</h2>
			<div className="flex max-sm:flex-col sm:items-center gap-7 mb-8">
				<Button>
					<a href="/util/time-calculator">时间计算器</a>
				</Button>

				<div className="flex sm:w-96 gap-3">
					<Input placeholder="输入BV号或av号" value={input} onChange={(e) => setInput(e.target.value)} />
					<Button>
						<a href={`/song/${input}/add`}>收录视频</a>
					</Button>
				</div>
			</div>

			<h2 className="text-2xl mb-4">即将达成成就</h2>
			<div className="flex items-center gap-4 mb-6">
				<Select value={milestoneType} onValueChange={(value: MilestoneType) => setMilestoneType(value)}>
					<SelectTrigger className="w-20">
						<SelectValue placeholder="成就" />
					</SelectTrigger>
					<SelectContent>
						<SelectItem value="dendou">殿堂</SelectItem>
						<SelectItem value="densetsu">传说</SelectItem>
						<SelectItem value="shinwa">神话</SelectItem>
					</SelectContent>
				</Select>
				<span className="text-xs text-muted-foreground">
					播放量在 {milestoneConfig[milestoneType].range[0].toLocaleString()} -{" "}
					{milestoneConfig[milestoneType].range[1].toLocaleString()} 之间，即将达成
					{milestoneConfig[milestoneType].name}
				</span>
			</div>

			<MilestoneVideos />
		</Layout>
	);
}
