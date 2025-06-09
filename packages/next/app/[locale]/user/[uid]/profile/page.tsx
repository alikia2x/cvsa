import { getUserProfile, User } from "@/lib/userAuth";
import { notFound } from "next/navigation";
import { format } from "date-fns";
import { zhCN } from "date-fns/locale";
import { LogoutButton } from "./LogoutButton";
import { numeric } from "yup-numeric";
import { getTranslations } from "next-intl/server";
import HeaderServer from "@/components/shell/HeaderServer";
import { Content } from "@/components/shell/Content";
import { ContentClient } from "@/components/shell/ContentClient";

const uidSchema = numeric().integer().min(0);

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

export default async function ProfilePage({ params }: { params: Promise<{ uid: string }> }) {
	const { uid } = await params;
	const t = await getTranslations("profile_page");
	let parsedUID: number;

	try {
		uidSchema.validate(uid);
		parsedUID = parseInt(uid);
	} catch (error) {
		return notFound();
	}

	const user = await getUserProfile(parsedUID);

	if (!user) {
		return notFound();
	}

	const displayName = user.nickname || user.username;
	const loggedIn = user.isLoggedIn;

	return (
		<>
			<HeaderServer />
			<main className="px-4 md:w-xl lg:w-3xl xl:w-4xl md:mx-auto pt-6 content-box mb-8">
				<h1 className="relative w-full">
					<span className="text-4xl font-extralight">{displayName}</span>
					<span className="ml-2 text-on-surface-variant dark:text-dark-on-surface-variant">
						UID{user.uid}
					</span>
					<div className="absolute right-0 top-0">{loggedIn && <LogoutButton />}</div>
				</h1>
				<SignupTime user={user} />
				<p>权限组：{t(`role.${user.role}`)}</p>

				<h2 className="mt-4 text-2xl">个人简介</h2>
				<Content pageID={`user-profile:${uid}`} />
			</main>
		</>
	);
}
