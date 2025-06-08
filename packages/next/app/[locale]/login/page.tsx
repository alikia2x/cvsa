import { LeftArrow } from "@/components/icons/LeftArrow";
import { RightArrow } from "@/components/icons/RightArrow";
import LoginForm from "./LoginForm";
import { Link, redirect } from "@/i18n/navigation";
import { getLocale } from "next-intl/server";
import { getCurrentUser } from "@/lib/userAuth";

export default async function LoginPage() {
	const user = await getCurrentUser();
	const locale = await getLocale();

	if (user) {
		redirect({
			href: `/user/${user.uid}/profile`,
			locale: locale
		});
	}
	return (
		<main className="relative flex-grow pt-8 px-4 md:w-full md:h-full md:flex md:items-center md:justify-center">
			<div
				className="md:w-[40rem] rounded-md md:p-8 md:-translate-y-6
				        md:bg-surface-container md:dark:bg-dark-surface-container"
			>
				<p className="mb-2">
					<Link href="/">
						<LeftArrow className="inline -translate-y-0.5 scale-90 mr-1" aria-hidden="true" />
						首页
					</Link>
				</p>
				<h1 className="text-5xl leading-[4rem] font-extralight">登录</h1>
				<p className="mt-4 mb-6">
					没有账户？
					<Link href="/singup">
						<span>注册</span>
						<RightArrow className="text-xs inline -translate-y-0.5 ml-1" aria-hidden="true" />
					</Link>
				</p>
				<LoginForm backendURL={process.env.NEXT_PUBLIC_BACKEND_URL ?? ""} />
			</div>
		</main>
	);
}
