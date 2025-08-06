import { Title } from "@solidjs/meta";
import { HttpStatusCode } from "@solidjs/start";
import { Layout } from "~/components/layout";
import { A } from "@solidjs/router";

export default function NotFound() {
	return (
		<Layout>
			<Title>找不到页面啾～</Title>
			<HttpStatusCode code={404} />
			<main class="w-full h-[calc(100vh-6rem)] flex flex-col flex-grow items-center justify-center gap-8">
				<h1 class="text-9xl font-thin">404</h1>
				<p class="text-xl font-medium">咦……页面去哪里了(ﾟДﾟ≡ﾟдﾟ)!?</p>
				<A href="/">
					带我回首页！
				</A>
			</main>
		</Layout>
	);
}
