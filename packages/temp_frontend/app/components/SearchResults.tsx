import { z } from "zod";
import type { SearchResult } from "@/routes/search";

interface SearchResultsProps {
	results: SearchResult;
	query: string;
}

export const formatDateTime = (date: Date, showYear = true, showSec = false): string => {
	const year = date.getFullYear();
	const month = String(date.getMonth() + 1).padStart(2, "0"); // 月份从0开始，补0
	const day = String(date.getDate()).padStart(2, "0");
	const hour = String(date.getHours()).padStart(2, "0");
	const minute = String(date.getMinutes()).padStart(2, "0");
	const second = String(date.getSeconds()).padStart(2, "0");
	const yearStr = showYear ? ` ${year}-` : "";
	const secStr = showSec ? `:${second}` : "";
	return `${yearStr}${month}-${day} ${hour}:${minute}${secStr}`;
};

const biliIDSchema = z.union([z.string().regex(/BV1[0-9A-Za-z]{9}/), z.string().regex(/av[0-9]+/)]);

export function SearchResults({ results, query }: SearchResultsProps) {
	if (!results || results.data.length === 0) {
		if (!biliIDSchema.safeParse(query).success) {
			return (
				<div className="text-center pt-6">
					<p className="text-secondary-foreground">没有找到相关结果</p>
				</div>
			);
		}
		return (
			<div className="text-center pt-6">
				<p className="text-secondary-foreground">
					没有找到相关结果。 尝试
					<a href={`/song/${query}/add`} className="text-primary">
						收录
					</a>
					?
				</p>
			</div>
		);
	}

	const SearchResultItem = ({
		result,
	}: {
		result: Exclude<SearchResult, null>["data"][number];
	}) => {
		switch (result.type) {
			case "song":
				return <SongResult result={result} />;
			case "bili-video":
				return <BiliVideoResult result={result} />;
			case "bili-video-db":
				return <BiliVideoDBResult result={result} />;
			default:
				return null;
		}
	};

	return (
		<div className="space-y-4 mb-20 mt-5">
			<p>
				找到 {results.data.length} 个结果（{(results.elapsedMs / 1000).toFixed(3)}秒）：
			</p>
			{results.data.map((result, index) => (
				<div
					// biome-ignore lint/suspicious/noArrayIndexKey: `result` will never change
					key={index}
					className="bg-white dark:bg-neutral-800 rounded-lg shadow-sm border
					 border-gray-200 dark:border-neutral-700 p-2 sm:p-4 hover:shadow-md transition-shadow"
				>
					<SearchResultItem result={result} />
				</div>
			))}
		</div>
	);
}

function SongResult({ result }: { result: Exclude<SearchResult, null>["data"][number] }) {
	if (result.type !== "song") return null;

	const { data } = result;
	return (
		<div className="flex items-center space-x-4">
			{data.image && (
				<img
					src={data.image}
					alt="歌曲封面"
					className="h-21 w-36 sm:w-42 sm:h-24 rounded-sm object-cover shrink-0"
					referrerPolicy="no-referrer"
				/>
			)}
			<div className="flex-col">
				<h3 className="text-lg font-medium line-clamp-1 text-wrap">{data.name}</h3>
				{data.producer && (
					<p className="text-sm text-muted-foreground truncate">{data.producer}</p>
				)}
				<div className="flex items-center space-x-4 my-1 text-xs text-muted-foreground">
					{data.duration && (
						<span>
							{Math.floor(data.duration / 60)}:
							{(data.duration % 60).toString().padStart(2, "0")}
						</span>
					)}
					{data.publishedAt && <span>{formatDateTime(new Date(data.publishedAt))}</span>}
				</div>
				<div className="flex gap-2">
					{data.aid && (
						<a
							href={`https://www.bilibili.com/video/av${data.aid}`}
							className="text-pink-400 text-sm"
						>
							观看视频
						</a>
					)}
					<a href={`/song/${data.id}/info`} className="text-sm text-secondary-foreground">
						查看曲目详情
					</a>
				</div>
			</div>
		</div>
	);
}

function BiliVideoResult({ result }: { result: Exclude<SearchResult, null>["data"][number] }) {
	if (result.type !== "bili-video") return null;
	const { data } = result;
	return (
		<div className="flex flex-col items-start space-x-4">
			{data.pic && (
				<img
					src={data.pic}
					alt="视频封面"
					className="w-full rounded-lg object-cover"
					referrerPolicy="no-referrer"
				/>
			)}
			<div className="flex-col mt-4">
				<h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
					{data.title}
				</h3>
				{data.desc && (
					<pre className="text-sm font-sans text-gray-600 dark:text-gray-400 line-clamp-3 mt-1 text-wrap">
						{data.desc}
					</pre>
				)}
				<div className="grid-cols-2 sm:flex items-center space-x-4 my-2 text-xs text-gray-500 dark:text-gray-500">
					{data.duration && (
						<span>
							{Math.floor(data.duration / 60)}:
							{(data.duration % 60).toString().padStart(2, "0")}
						</span>
					)}
					{data.pubdate && <span>{formatDateTime(new Date(data.pubdate * 1000))}</span>}
					{data.bvid && <span>{data.bvid}</span>}
					{data.stat.view && <span>{data.stat.view.toLocaleString()} 播放</span>}
					{data.tname && <span>{data.tname}</span>}
				</div>
				<div className="flex gap-2">
					{data.bvid && (
						<a
							href={`https://www.bilibili.com/video/${data.bvid}`}
							target="_blank"
							rel="noopener noreferrer"
							className="text-pink-400 text-sm"
						>
							观看视频
						</a>
					)}
					{data.bvid && (
						<a
							href={`/video/av${data.aid}/info`}
							className="text-sm text-secondary-foreground"
						>
							查看视频详情
						</a>
					)}
				</div>
			</div>
		</div>
	);
}

function BiliVideoDBResult({ result }: { result: Exclude<SearchResult, null>["data"][number] }) {
	if (result.type !== "bili-video-db") return null;

	const { data } = result;
	return (
		<div className="flex flex-col items-start space-x-4">
			{data.coverUrl && (
				<img
					src={data.coverUrl}
					alt="视频封面"
					className="w-full rounded-lg object-cover"
					referrerPolicy="no-referrer"
				/>
			)}
			<div className="flex-col mt-4">
				<h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
					{data.title}
				</h3>
				{data.description && (
					<pre className="text-sm font-sans text-gray-600 dark:text-gray-400 line-clamp-3 mt-1 text-wrap">
						{data.description}
					</pre>
				)}
				<div className="grid-cols-2 sm:flex items-center space-x-4 my-2 text-xs text-gray-500 dark:text-gray-500">
					{data.duration && (
						<span>
							{Math.floor(data.duration / 60)}:
							{(data.duration % 60).toString().padStart(2, "0")}
						</span>
					)}
					{data.publishedAt && <span>{formatDateTime(new Date(data.publishedAt))}</span>}
					{data.bvid && <span>{data.bvid}</span>}
					{data.views && <span>{data.views.toLocaleString()} 播放</span>}
				</div>
				<div className="flex gap-2">
					{data.bvid && (
						<a
							href={`https://www.bilibili.com/video/${data.bvid}`}
							target="_blank"
							rel="noopener noreferrer"
							className="text-pink-400 text-sm"
						>
							观看视频
						</a>
					)}
					{data.bvid && (
						<a
							href={`/song/av${data.aid}/info`}
							className="text-sm text-secondary-foreground"
						>
							查看曲目详情
						</a>
					)}
				</div>
			</div>
		</div>
	);
}
