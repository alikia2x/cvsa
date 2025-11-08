import { Layout } from "@/components/Layout";
import type { Route } from "./+types/home";

export function meta({}: Route.MetaArgs) {
	return [{ title: "中V档案馆" }];
}

export default function Home() {
	return <Layout></Layout>;
}
