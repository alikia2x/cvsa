import type { Route } from "./+types/home";
import { treaty } from "@elysiajs/eden";
import type { App } from "@elysia/src";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SearchIcon } from "@/components/icons/search";

const app = treaty<App>("localhost:15412");

export function meta({}: Route.MetaArgs) {
	return [{ title: "中V档案馆" }];
}

export default function Home() {
	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<main className="w-full max-md:mx-6 pt-14 md:w-xl xl:w-2xl">
				<h1 className="text-4xl my-2">中V档案馆</h1>
				<div className="flex h-12 mt-5 gap-2 relative">
					<Input className="h-full pl-5 pr-12 rounded-full" type="search" placeholder="搜索" />
					<Button variant="ghost" className="absolute rounded-full size-10 top-1 right-1">
						<SearchIcon className="size-6" />
					</Button>
				</div>

				<Button
					className="w-full mt-4 h-10"
					variant="secondary"
					onClick={async () => {
						try {
							const { data } = await app.ping.get("");
							if (data && data.message === "pong") {
								toast("pong");
								return;
							}
							toast("校验失败。");
						} catch (e) {
							console.error(e);
							toast("发生错误。");
						}
					}}
				>
					Ping
				</Button>
			</main>
		</div>
	);
}
