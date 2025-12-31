import type { App } from "@backend/src";
import { treaty } from "@elysiajs/eden";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MilestoneVideoCard } from "./MilestoneVideoCard";

const app = treaty<App>(import.meta.env.VITE_API_URL!);

export type CloseMilestoneInfo = Awaited<
	ReturnType<ReturnType<(typeof app.songs)["close-milestone"]>["get"]>
>["data"];
type CloseMilestoneError = Awaited<
	ReturnType<ReturnType<(typeof app.songs)["close-milestone"]>["get"]>
>["error"];

export type MilestoneType = "dendou" | "densetsu" | "shinwa";

export const milestoneConfig: Record<
	MilestoneType,
	{ name: string; range: [number, number]; target: number }
> = {
	dendou: { name: "殿堂", range: [90000, 99999], target: 100000 },
	densetsu: { name: "传说", range: [900000, 999999], target: 1000000 },
	shinwa: { name: "神话", range: [5000000, 9999999], target: 10000000 },
};

export const MilestoneVideos: React.FC = () => {
	const [milestoneType, setMilestoneType] = useState<MilestoneType>("shinwa");
	const [milestoneData, setMilestoneData] = useState<CloseMilestoneInfo>([]);
	const [closeMilestoneError, setCloseMilestoneError] = useState<CloseMilestoneError>();
	const [isLoading, setIsLoading] = useState(false);
	const [isLoadingMore, setIsLoadingMore] = useState(false);
	const [offset, setOffset] = useState(0);
	const [hasMore, setHasMore] = useState(true);

	const scrollContainer = useRef<HTMLDivElement>(null);

	const fetchMilestoneData = useCallback(
		async (type: MilestoneType, reset: boolean = false) => {
			const currentOffset = reset ? 0 : offset;

			if (!reset) {
				setIsLoadingMore(true);
			} else {
				setIsLoading(true);
			}

			setCloseMilestoneError(undefined);

			try {
				const { data, error } = await app.songs["close-milestone"]({ type }).get({
					query: {
						limit: 20,
						offset: currentOffset,
					},
				});

				if (error) {
					setCloseMilestoneError(error);
				} else {
					if (reset) {
						setMilestoneData(data);
					} else {
						setMilestoneData((prev) => [...prev!, ...data]);
					}
					setHasMore(data.length >= 20);
				}
			} catch (err) {
				console.error("Fetch error:", err);
			} finally {
				setIsLoading(false);
				setIsLoadingMore(false);
			}
		},
		[offset]
	);

	useEffect(() => {
		setOffset(0);
		setHasMore(true);
		setMilestoneData([]);
		fetchMilestoneData(milestoneType, true);
	}, [milestoneType]);

	useEffect(() => {
		if (offset > 0 && hasMore && !isLoadingMore) {
			fetchMilestoneData(milestoneType);
		}
	}, [offset]);

	const handleScroll = useCallback(
		(e: React.UIEvent<HTMLDivElement>) => {
			const target = e.currentTarget;
			const { scrollHeight, scrollTop, clientHeight } = target;

			if (scrollTop + clientHeight >= scrollHeight - 500 && !isLoadingMore && hasMore) {
				setOffset((prev) => prev + 20);
			}
		},
		[hasMore, isLoadingMore]
	);

	const renderContent = () => {
		if (!milestoneData) return null;

		if (isLoading && milestoneData.length === 0) {
			return (
				<ScrollArea className="h-140 xl:h-180 w-full">
					<div className="h-[0.1px]"></div>
					{[1, 2, 3].map((i) => (
						<div
							key={i}
							className="rounded-xl my-4 shadow-sm border border-gray-200 dark:border-neutral-700"
						>
							<Skeleton className="h-49 sm:h-55 rounded-xl" />
						</div>
					))}
				</ScrollArea>
			);
		}

		if (closeMilestoneError && milestoneData.length === 0) {
			return (
				<div className="text-center py-8">
					<p className="text-red-500">
						加载失败: {closeMilestoneError.value?.message || "未知错误"}
					</p>
					<Button
						variant="outline"
						className="mt-4"
						onClick={() => fetchMilestoneData(milestoneType, true)}
					>
						重试
					</Button>
				</div>
			);
		}

		if (milestoneData.length === 0) {
			return (
				<div className="text-center py-8">
					<p className="text-secondary-foreground">
						暂无接近{milestoneConfig[milestoneType].name}的视频
					</p>
				</div>
			);
		}

		return (
			<div className="space-y-4">
				<ScrollArea
					className="h-140 xl:h-180 w-full"
					ref={scrollContainer}
					onScroll={handleScroll}
				>
					{milestoneData.map((video) => (
						<MilestoneVideoCard
							key={video.bilibili_metadata.aid}
							video={video}
							milestoneType={milestoneType}
						/>
					))}
					{isLoadingMore && (
						<div className="rounded-xl my-4 shadow-sm border border-gray-200 dark:border-neutral-700">
							<Skeleton className="h-49 sm:h-55 w-full rounded-xl" />
						</div>
					)}
				</ScrollArea>
			</div>
		);
	};

	return (
		<>
			<div className="flex justify-between mt-6 mb-2">
				<h2 className="text-2xl font-medium">成就助攻</h2>
				<Tabs
					value={milestoneType}
					onValueChange={(value: string) => setMilestoneType(value as MilestoneType)}
				>
					<TabsList>
						<TabsTrigger value="dendou">殿堂</TabsTrigger>
						<TabsTrigger value="densetsu">传说</TabsTrigger>
						<TabsTrigger value="shinwa">神话</TabsTrigger>
					</TabsList>
				</Tabs>
			</div>

			{renderContent()}
		</>
	);
};
