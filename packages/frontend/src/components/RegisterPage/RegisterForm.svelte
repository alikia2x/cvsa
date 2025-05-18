<script lang="ts">
	import TextField from "@components/TextField.svelte";
	import LoadingSpinner from "@components/icon/LoadingSpinner.svelte";
	import { computeVdfInWorker } from "@lib/vdf.js";

	export let backendURL: string;

	let password = '';
	let username = '';
	let nickname = '';

	let loading = false;

	async function createCaptchaSession() {
		const url = new URL(backendURL);
		url.pathname = '/captcha/session';
		const res = await fetch(url.toString(), {
			method: "POST",
			body: JSON.stringify({
				"route": "POST-/user"
			})
		});
		if (res.status !== 201) {
			throw new Error("Failed to create captcha session");
		}
		return await res.json();
	}

	async function getCaptchaResult(id: string, ans: string) {
		const url = new URL(backendURL);
		url.pathname = `/captcha/${id}/result`;
		url.searchParams.set("ans", ans);
		const res = await fetch(url.toString());
		if (res.status !== 200) {
			throw new Error("Failed to verify captcha answer");
		}
		return await res.json();
	}

	async function register() {
		const { g, n, t, id } = await createCaptchaSession();
		const ans = await computeVdfInWorker(BigInt(g), BigInt(n), BigInt(t));
		const res = await getCaptchaResult(id, ans.result.toString());
		console.log(res)
	}
</script>

<form class="w-full flex flex-col gap-6">
	<TextField labelText="用户名" bind:inputText={username} maxChar={50}
			   supportingText="*必填。用户名是唯一的，不区分大小写。"
	/>
	<TextField labelText="密码" type="password" bind:inputText={password}
			   supportingText="*必填。密码至少为 4 个字符。" maxChar={120}
	/>
	<TextField labelText="昵称" bind:inputText={nickname}
			   supportingText="昵称可以重复。" maxChar={30}
	/>
	<button class="bg-primary dark:bg-dark-primary text-on-primary dark:text-dark-on-primary duration-150
		rounded-full hover:bg-on-primary-container hover:dark:bg-dark-on-primary-container mt-2
		flex items-center text-sm leading-5 justify-center h-10 w-full"
		onclick={async (e) => {
			e.preventDefault();
			loading = true;
			try {
				await register();
			}
			finally {
			    loading = false;
			}
		}}
	>
		{#if !loading}
			<span>注册</span>
		{:else}
			<LoadingSpinner/>
		{/if}
	</button>
</form>