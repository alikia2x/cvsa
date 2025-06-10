import { getUserProfile, User } from "@/lib/userAuth";
import { notFound } from "next/navigation";
import { format } from "date-fns";
import { zhCN } from "date-fns/locale";
import { LogoutButton } from "./LogoutButton";
import { numeric } from "yup-numeric";
import { getTranslations } from "next-intl/server";
import HeaderServer from "@/components/shell/HeaderServer";
import { DateTime } from "luxon";

const uidSchema = numeric().integer().min(0);

interface SignupTimeProps {
	user: User;
}

const SignupTime: React.FC<SignupTimeProps> = ({ user }: SignupTimeProps) => {
	return (
		<p className="mt-4">
			于&nbsp;
			{DateTime.fromJSDate(user.createdAt).toFormat("yyyy-MM-dd HH:mm:ss")}
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
			<main className="md:w-xl lg:w-2xl xl:w-3xl md:mx-auto pt-6">
				<h1>
					<span className="text-4xl font-extralight">{displayName}</span>
					<span className="ml-2 text-on-surface-variant dark:text-dark-on-surface-variant">
						UID{user.uid}
					</span>
				</h1>
				<SignupTime user={user} />
				<p className="mt-4">权限组：{t(`role.${user.role}`)}</p>
				{loggedIn && <LogoutButton />}
			</main>
		</>
	);
}
