import { toast } from "sonner";
import { Search } from "./Search";

export function Layout({ children }: { children?: React.ReactNode }) {
	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<main className="w-full max-sm:mx-3 pt-14 sm:w-xl xl:w-2xl">
				<div className="flex items-center justify-between">
					<a href="/">
						<h1 className="text-3xl mb-5">中V档案馆</h1>
					</a>
					<div className="h-8">
						<LoginOrLogout />
					</div>
				</div>
				<Search />
				{children}
			</main>
		</div>
	);
}

const LoginOrLogout = () => {
	const session = localStorage.getItem("sessionID");
	if (session) {
		return (
			<span
				onClick={() => {
					localStorage.removeItem("sessionID");
					toast.success("已退出登录");
				}}
			>
				退出登录
			</span>
		);
	} else {
		return <a href="/login">登录</a>;
	}
};

export function LayoutWithoutSearch({ children }: { children?: React.ReactNode }) {
	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<main className="w-full max-sm:mx-3 pt-14 sm:w-xl xl:w-2xl">
				<div className="flex items-center justify-between">
					<a href="/">
						<h1 className="text-3xl mb-5">中V档案馆</h1>
					</a>
					<div className="h-8">
						<LoginOrLogout />
					</div>
				</div>

				{children}
			</main>
		</div>
	);
}
