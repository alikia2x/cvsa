import { TriangleAlert } from "lucide-react";
import { Title } from "./Title";

export function Error({ error }: { error: { status: number; value: { message?: string } } }) {
	return (
		<div className="w-screen min-h-screen flex items-center justify-center">
			<Title title="出错了" />
			<div className="max-w-md w-full mx-4 bg-gray-100 dark:bg-neutral-900 rounded-2xl 
				shadow-lg p-6 flex flex-col gap-4 items-center text-center">
				<div className="w-16 h-16 flex items-center justify-center rounded-full bg-red-500 text-white text-3xl">
					<TriangleAlert size={34} className="-translate-y-0.5" />
				</div>
				<h1 className="text-3xl font-semibold text-neutral-900 dark:text-neutral-100">出错了</h1>
				<p className="text-neutral-700 dark:text-neutral-300">状态码：{error.status}</p>
				{error.value.message && (
					<p className="text-neutral-600 dark:text-neutral-400 break-words">
						<span className="font-medium text-neutral-700 dark:text-neutral-300">错误信息</span>
						<br />
						{error.value.message}
					</p>
				)}
			</div>
		</div>
	);
}
