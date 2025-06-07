import HeaderServer from "@/components/shell/HeaderServer";
import tpLicense from "@/content/THIRD-PARTY-LICENSES.txt";
import projectLicense from "@/content/LICENSE.txt";

export default function LicensePage() {
	return (
		<>
			<HeaderServer />
			<main className="lg:max-w-4xl lg:mx-auto">
				<p className="leading-10">中 V 档案馆的软件在 AGPL 3.0 下许可，请见：</p>
				<pre className="break-all whitespace-pre-wrap">{projectLicense}</pre>
				<p className="leading-10">本项目引入的其它项目项目的许可详情如下：</p>
				<pre className="break-all whitespace-pre-wrap">{tpLicense}</pre>
			</main>
		</>
	);
}
