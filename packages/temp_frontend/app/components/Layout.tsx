import { Search } from "./Search";

export function Layout({ children }: { children?: React.ReactNode }) {
	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<main className="w-full max-sm:mx-6 pt-14 sm:w-xl xl:w-2xl">
				<a href="/">
					<h1 className="text-4xl mb-5">中V档案馆</h1>
				</a>
				<Search />
				{children}
			</main>
		</div>
	);
}

export function LayoutWithouSearch({ children }: { children?: React.ReactNode }) {
	return (
		<div className="w-screen min-h-screen relative left-0 top-0 flex justify-center">
			<main className="w-full max-sm:mx-6 pt-14 sm:w-xl xl:w-2xl">
				<a href="/">
					<h1 className="text-4xl mb-5">中V档案馆</h1>
				</a>
				{children}
			</main>
		</div>
	);
}
