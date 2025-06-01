import { Header } from "@/components/shell/Header";
import { getCurrentUser } from "@/lib/userAuth";
import { redirect } from "next/navigation";

export default async function ProfilePage() {
	const user = await getCurrentUser();

	if (!user) {
		redirect("/login");
	}

	return (
		<>
			<Header user={user} />
		</>
	);
}
