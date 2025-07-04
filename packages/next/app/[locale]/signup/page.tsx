import { LeftArrow } from "@/components/icons/LeftArrow";
import { RightArrow } from "@/components/icons/RightArrow";
import SignUpForm from "./SignUpForm";
import { Link } from "@/i18n/navigation";

export default function SignupPage() {
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
				<h1 className="text-5xl leading-[4rem] font-extralight">欢迎</h1>
				<p className="mt-2 md:mt-3">
					欢迎来到中 V 档案馆。
					<br />
					这里是中文虚拟歌手相关信息的收集站与档案馆。
				</p>
				<p className="my-2">
					注册一个账号，
					<br />
					让我们一起见证中 V 的历史，现在，与未来。
				</p>
				<p className="mt-4 mb-7">
					已有账户？
					<Link href="/login">
						<span>登录</span>
						<RightArrow className="text-xs inline -translate-y-0.5 ml-1" aria-hidden="true" />
					</Link>
				</p>
				<SignUpForm backendURL={process.env.NEXT_PUBLIC_BACKEND_URL ?? ""} />
			</div>
		</main>
	);
}
