<script lang="ts">
	import SearchIcon from "src/components/icon/SearchIcon.svelte";
	import CloseIcon from "src/components/icon/CloseIcon.svelte";

	let inputBox: HTMLInputElement | null = null;
	export let close = () => {};

	export function changeFocusState(target: boolean) {
		if (!inputBox) return;
		if (target) {
			inputBox.focus();
		} else {
			inputBox.blur();
		}
	}

	function search(query: string) {
		window.location.href = `/song/${query}/info`;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === "Enter") {
			event.preventDefault();
			const input = event.target as HTMLInputElement;
			const value = input.value.trim();
			if (!value) return;
			search(value);
		}
	}
</script>

<style>
    [type="search"]::-webkit-search-cancel-button,
    [type="search"]::-webkit-search-decoration {
        -webkit-appearance: none;
        appearance: none;
    }
</style>

<!-- svelte-ignore a11y_autofocus -->
<div class="absolute md:relative left-0 h-full mr-0 inline-flex items-center w-full px-4 md:px-0
	md:w-full xl:max-w-[50rem] md:mx-4">
	<div class="w-full h-10 lg:h-12 px-4 rounded-full bg-surface-container-high dark:bg-zinc-800/70
			backdrop-blur-lg flex justify-between md:px-5">
		<button class="w-6" on:click={() => search(inputBox?.value ?? "")}>
			<SearchIcon className="h-full inline-flex items-center text-[1.5rem]
			text-on-surface-variant dark:text-dark-on-surface-variant" />
		</button>
		<!--suppress HtmlUnknownAttribute -->
		<input
				bind:this={inputBox}
				type="search"
				placeholder="搜索"
				autocomplete="off"
				autocapitalize="off"
				autocorrect="off"
				class="top-0 h-full bg-transparent flex-grow px-4 focus:outline-none"
				on:keydown={handleKeydown}
		/>
		<button class="w-6" on:click={() => {inputBox.value = ""; close();}}>
			<CloseIcon className="h-full w-6 inline-flex items-center text-[1.5rem]
			text-on-surface-variant dark:text-dark-on-surface-variant"/>
		</button>
	</div>
</div>
