import { Header } from "@/components/shell/Header";
import { getCurrentUser } from "@/lib/userAuth";

export default async function HeaderServer() {
	const user = await getCurrentUser();
	return <Header user={user} />;
}
