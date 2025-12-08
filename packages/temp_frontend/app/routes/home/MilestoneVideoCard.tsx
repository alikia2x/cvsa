import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatDateTime } from "@/components/SearchResults";
import { addHoursToNow, formatHours } from "../song/[id]/info";
import { milestoneConfig, type CloseMilestoneInfo, type MilestoneType } from "./Milestone";
import { DAY, HOUR, MINUTE, SECOND } from "@core/lib";

function timeAgo(timeStamp: Date | number, now: Date | number = Date.now()): string {
	const pastTime = typeof timeStamp === "number" ? timeStamp : timeStamp.getTime();
	const currentTime = typeof now === "number" ? now : now.getTime();

	const diffMs = Math.abs(currentTime - pastTime);

	if (diffMs < MINUTE) {
		const seconds = Math.floor(diffMs / SECOND);
		return `${seconds} 秒`;
	}

	if (diffMs < HOUR) {
		const minutes = Math.floor(diffMs / MINUTE);
		if (diffMs % MINUTE === 0) {
			return `${minutes} 分钟`;
		}
		return `${minutes} 分钟`;
	}

	if (diffMs < DAY) {
		const hours = Math.floor(diffMs / HOUR);
		const minutes = Math.floor((diffMs % HOUR) / MINUTE);
		return `${hours} 时 ${minutes} 分`;
	}

	if (diffMs >= DAY) {
		const days = Math.floor(diffMs / DAY);
		const hours = Math.floor((diffMs % DAY) / HOUR);
		return `${days} 天 ${hours} 时`;
	}

	return "刚刚";
}

export const MilestoneVideoCard = ({
	video,
	milestoneType,
}: {
	video: NonNullable<CloseMilestoneInfo>[number];
	milestoneType: MilestoneType;
}) => {
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
				<span>数据更新于 {timeAgo(new Date(video.eta.updatedAt))}前</span>
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
