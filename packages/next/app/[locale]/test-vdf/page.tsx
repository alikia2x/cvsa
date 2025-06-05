import { Header } from "@/components/shell/Header";
import { getCurrentUser } from "@/lib/userAuth";
import { VDFtestCard } from "./TestCard";

export default async function VdfBenchmarkPage() {
	const user = await getCurrentUser();

	return (
		<>
			<Header user={user} />
			<div className="md:w-2/3 lg:w-1/2 xl:w-[37%] md:mx-auto mx-6 mb-12">
				<VDFtestCard />
				<div>
					<h2 className="text-xl font-medium leading-10">关于本页</h2>
					<div className="text-on-surface-variant dark:text-dark-on-surface-variant">
						<p>
							这是一个性能测试页面，
							<br />
							旨在测试我们的一个 VDF (Verifiable Delayed Function, 可验证延迟函数) 实现的性能。
							<br />
							这是一个数学函数，它驱动了整个网站的验证码（CAPTCHA）。
							<br />
							通过使用该函数，我们可以让您无需通过点选图片或滑动滑块既可完成验证，
							同时防御我们的网站，使其免受自动程序的攻击。
							<br />
						</p>
						<p>
							点击按钮，会自动测试并展示结果。
							<br />
						</p>
						<p>
							你可以将结果发送至邮箱：
							<a href="mailto:contact@alikia2x.com">contact@alikia2x.com</a> 或 QQ：
							<a href="https://qm.qq.com/q/WS8zyhlcEU">1559913735</a>，并附上自己的设备信息
							（例如，手机型号、电脑的 CPU 型号等）。
							<br />
							我们会根据测试结果，优化我们的实现，使性能更优。
							<br />
							感谢你的支持！
							<br />
						</p>
					</div>
				</div>
			</div>
		</>
	);
}
