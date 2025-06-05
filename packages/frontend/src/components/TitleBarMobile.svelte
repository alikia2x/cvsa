<script lang="ts">
	import logoMobileDark from "@assets/TitleBar Mobile Dark.svg";
	import logoMobileLight from "@assets/TitleBar Mobile Light.svg";
	import SearchBox from "@components/SearchBox.svelte";
	import SearchIcon from "src/components/icon/SearchIcon.svelte";
	import MenuIcon from "@components/icon/MenuIcon.svelte";
	import DarkModeImage from "@components/DarkModeImage.svelte";
	import NavigationDrawer from "@components/NavigationDrawer.svelte";
	import HomeIcon from "@components/icon/HomeIcon.svelte";
	import InfoIcon from "@components/icon/InfoIcon.svelte";
	import RegisterIcon from "@components/icon/RegisterIcon.svelte";
	import Portal from "@components/Portal.svelte";

	let searchBox: SearchBox | null = null;
	let showSearchBox = false;
	let showDrawer = false;

	$: if (showSearchBox && searchBox) {
		searchBox.changeFocusState(true);
	}
</script>

<Portal>
	<NavigationDrawer show={showDrawer} onClose={() => showDrawer = false}>
		<div class="flex flex-col w-full">
			<div class="w-full h-14 flex items-center px-4">
				<HomeIcon className="text-2xl pr-4"/>
				<a href="/">首页</a>
			</div>
			<div class="w-full h-14 flex items-center px-4">
				<InfoIcon className="text-2xl pr-4"/>
				<a href="/about">关于</a>
			</div>
			<div class="w-full h-14 flex items-center px-4">
				<RegisterIcon className="text-2xl pr-4"/>
				<a href="/register">注册</a>
			</div>
		</div>
	</NavigationDrawer>
</Portal>

<div class="md:hidden relative top-0 left-0 w-full h-16 z-20">
	{#if !showSearchBox}
		<button class="inline-flex absolute left-0 ml-4 h-full items-center dark:text-white"
				onclick={() => showDrawer = true}
		>
			<MenuIcon/>
		</button>
		<div class="absolute left-1/2 -translate-x-1/2 -translate-y-0.5 inline-flex h-full items-center">
			<a href="/">
				<DarkModeImage
						lightSrc={logoMobileLight.src}
						darkSrc={logoMobileDark.src}
						alt="Logo"
						className="w-24 h-8 translate-y-[2px]"
				/>
			</a>
		</div>
	{/if}
	{#if showSearchBox}
		<SearchBox bind:this={searchBox} close={() => showSearchBox = false}/>
	{/if}
	{#if !showSearchBox}
		<button
				class="inline-flex absolute right-0 h-full items-center mr-4"
				onclick={() => (showSearchBox = !showSearchBox)}
		>
			<SearchIcon className="text-[1.625rem]"/>
		</button>
	{/if}
</div>
