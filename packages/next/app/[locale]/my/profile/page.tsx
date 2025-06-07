import { Header } from "@/components/shell/Header";
import { getCurrentUser, User } from "@/lib/userAuth";
import { redirect } from "next/navigation";
import { format } from "date-fns";
import { zhCN } from "date-fns/locale";

interface SignupTimeProps {
	user: User;
}

const SignupTime: React.FC<SignupTimeProps> = ({ user }: SignupTimeProps) => {
	return (
		<p className="mt-4">
			于&nbsp;
			{format(new Date(user.createdAt), "yyyy-MM-dd HH:mm:ss", {
				locale: zhCN
			})}
			&nbsp;注册。
		</p>
	);
};

export default async function ProfilePage() {
	const user = await getCurrentUser();

	if (!user) {
		redirect("/login");
	}

	const displayName = user.nickname || user.username;

	return (
		<>
			<Header user={user} />
			<main className="md:w-xl lg:w-2xl xl:w-3xl md:mx-auto pt-6">
				<h1>
					<span className="text-4xl font-extralight">{displayName}</span>
					<span className="ml-2 text-on-surface-variant dark:text-dark-on-surface-variant">
						UID{user.uid}
					</span>
				</h1>
				<SignupTime user={user} />
			</main>
		</>
	);
}
