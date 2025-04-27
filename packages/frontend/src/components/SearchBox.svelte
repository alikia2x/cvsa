<script lang="ts">
	import SearchIcon from "src/components/icon/SearchIcon.svelte";
	import CloseIcon from "src/components/icon/CloseIcon.svelte";

	let inputValue = ""; // 使用一个变量来绑定 input 的值
	export let close = () => {
	};

	export function changeFocusState(target: boolean) {
		if (!inputElement) return; // 使用 inputElement 而不是 inputBox
		if (target) {
			inputElement.focus();
		} else {
			inputElement.blur();
		}
	}

	function search(query: string) {
		window.location.href = `/song/${query}/info`;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === "Enter") {
			event.preventDefault();
			const value = inputValue.trim(); // 使用绑定的变量
			if (!value) return;
			search(value);
		}
	}

	let inputElement: HTMLInputElement; // 引用 input 元素
</script>

<style>
    [type="search"]::-webkit-search-cancel-button,
    [type="search"]::-webkit-search-decoration {
        -webkit-appearance: none;
        appearance: none;
    }
</style>

<div class="absolute md:relative left-0 h-full mr-0 inline-flex items-center w-full px-4 md:px-0
    md:w-full xl:max-w-[50rem] md:mx-4">
	<div class="w-full h-10 lg:h-12 px-4 rounded-full bg-surface-container-high dark:bg-zinc-800/70
          backdrop-blur-lg flex justify-between md:px-5">
		<button class="w-6" on:click={() => search(inputValue)}>
			<SearchIcon className="h-full inline-flex items-center text-[1.5rem]
          text-on-surface-variant dark:text-dark-on-surface-variant"/>
		</button>
		<input
				bind:this={inputElement}
				bind:value={inputValue}
				type="search"
				placeholder="搜索"
				autocomplete="off"
				autocapitalize="off"
				autocorrect="off"
				class="top-0 h-full bg-transparent flex-grow px-4 focus:outline-none"
				on:keydown={handleKeydown}
		/>
		<button class={"w-6 duration-100 " + (inputValue ? "md:opacity-100" : "md:opacity-0") } on:click={() => {inputValue = ""; close();}}>
			<CloseIcon className="h-full w-6 inline-flex items-center text-[1.5rem]
	  text-on-surface-variant dark:text-dark-on-surface-variant"/>
		</button>
	</div>
</div>