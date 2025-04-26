<script lang="ts">
	import {onMount} from 'svelte';
	import {fly} from 'svelte/transition';
	import {fade} from 'svelte/transition';

	export let show: boolean = false;
	export let onClose: () => void;

	let drawer: HTMLDivElement;
	let cover: HTMLDivElement;

	onMount(() => {
		const handleOutsideClick = (event: MouseEvent) => {
			if (show && event.target === cover) {
				onClose();
			}
		};

		window.addEventListener('click', handleOutsideClick);

		return () => {
			window.removeEventListener('click', handleOutsideClick);
		};
	});
</script>

<div class="absolute z-50 ">
	{#if show}
		<div
				bind:this={cover}
				transition:fade="{{ duration: 300 }}"
				class="fixed top-0 left-0 w-full h-full z-40 bg-[#00000020]"
				aria-hidden="true">

		</div>

		<div
				bind:this={drawer}
				transition:fly="{{ x: -500, duration: 300 }}" class="fixed top-0 left-0 h-full
			bg-[#fff0ee] dark:bg-[#231918] z-50"
				style="width: min(22.5rem, 70vw);"
				role="dialog" aria-modal="true">
			<slot></slot>
		</div>
	{/if}
</div>
