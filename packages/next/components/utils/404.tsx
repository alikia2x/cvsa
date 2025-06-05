import { getCurrentUser } from "@/lib/userAuth";
import { Header } from "@/components/shell/Header";
import { Link } from "@/i18n/navigation";
import { getTranslations } from "next-intl/server";

export const NotFound = async () => {
	const user = await getCurrentUser();
	const t = await getTranslations("not_found");
	return (
		<>
			<Header user={user} />

			<main className="flex flex-col flex-grow items-center justify-center gap-8">
				<h1 className="text-9xl font-thin">404</h1>
				<p className="text-xl font-medium">{t("title")}</p>
				<Link href="/">{t("back_to_home")}</Link>
			</main>
		</>
	);
};
