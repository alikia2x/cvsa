import { Header } from "@/components/shell/Header";
import { getCurrentUser } from "@/lib/userAuth";
import AboutContent from "./AboutContent";

export default async function AboutPage() {
	const user = await getCurrentUser();
	return (
		<>
			<Header user={user} />
			<main className="flex flex-col items-center min-h-screen gap-8 md:mt-12 relative z-0">
				<div className="w-full lg:w-2/3 xl:w-1/2 content px-8 md:px-12 lg:px-0">
					<AboutContent />
				</div>
			</main>
		</>
	);
}
