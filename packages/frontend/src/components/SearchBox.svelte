<script lang="ts">
    export let autoFocus = false;

    function handleKeydown(event: KeyboardEvent) {
        if (event.key === "Enter") {
            event.preventDefault();
            const input = event.target as HTMLInputElement;
            const value: string = input.value.trim();
            if (!value) return;
            if (value.startsWith("av")) {
                window.location.href = `/song/${value.slice(2)}/info`;
            }
        }
    }
</script>

<!-- svelte-ignore a11y_autofocus -->
<div
    class="absolute left-0 md:left-96 ml-4 w-[calc(100%-5rem)] md:w-[calc(100%-40rem)] 2xl:max-w-[50rem] 2xl:left-1/2 2xl:-translate-x-1/2 inline-flex items-center h-full"
    autofocus={autoFocus}
>
    <input
        type="search"
        placeholder="搜索"
        class="top-0 w-full h-10 px-4 rounded-lg bg-white/80 dark:bg-zinc-800/70
          backdrop-blur-lg border border-zinc-300 dark:border-zinc-600 focus:border-zinc-400 duration-200 transition-colors focus:outline-none"
        on:keydown={handleKeydown}
    />
</div>

<style>
    [type="search"]::-webkit-search-cancel-button,
    [type="search"]::-webkit-search-decoration {
        -webkit-appearance: none;
        appearance: none;
    }
</style>
