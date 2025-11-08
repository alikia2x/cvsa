import { LayoutWithoutSearch } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useNavigate } from "react-router";
import { treaty } from "@elysiajs/eden";
import type { App } from "@elysia/src";

// @ts-expect-error anyway...
const app = treaty<App>(import.meta.env.VITE_API_URL!);

export default function Login() {
	const navigate = useNavigate();
	const [formData, setFormData] = useState({
		username: "",
		password: "",
	});
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState("");

	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const { name, value } = e.target;
		setFormData((prev) => ({
			...prev,
			[name]: value,
		}));
		setError("");
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		setIsLoading(true);
		setError("");

		try {
			const { data, error } = await app.auth.session.post(formData);

			if (data) {
                localStorage.setItem("sessionID", data.sessionID);
				navigate("/", { replace: true });
			} else {
				setError(error?.value?.message || "登录失败");
			}
		} catch (err) {
			setError("网络错误，请稍后重试");
			console.error("Login error:", err);
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<LayoutWithoutSearch>
			<h2 className="text-2xl font-bold mb-6">登录</h2>

			{error && (
				<div className="mb-4 p-3 bg-red-100 dark:bg-red-900 border border-red-400 dark:border-red-600 text-red-700 dark:text-red-200 rounded">
					{error}
				</div>
			)}

			<form onSubmit={handleSubmit} className="space-y-4 flex flex-col">
				<div>
					<label htmlFor="username" className="block text-sm font-medium mb-2">
						用户名
					</label>
					<Input
						type="text"
						id="username"
						name="username"
						value={formData.username}
						onChange={handleChange}
						required
						placeholder="请输入用户名"
					/>
				</div>

				<div>
					<label htmlFor="password" className="block text-sm font-medium mb-2">
						密码
					</label>
					<Input
						type="password"
						id="password"
						name="password"
						value={formData.password}
						onChange={handleChange}
						required
						placeholder="请输入密码"
					/>
				</div>

				<Button type="submit" disabled={isLoading}>
					{isLoading ? "登录中..." : "登录"}
				</Button>
			</form>
		</LayoutWithoutSearch>
	);
}
