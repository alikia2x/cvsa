<script lang="ts">
    import { onMount } from "svelte";

    export let lightSrc: string;
    export let darkSrc: string;
    export let alt: string = "";
    export let className: string = "";

    let isDarkMode = false;
    let currentSrc: string;
    let opacity = 0;

    onMount(() => {
        const handleDarkModeChange = (event: MediaQueryListEvent) => {
            isDarkMode = event.matches;
            currentSrc = isDarkMode ? darkSrc : lightSrc;
            opacity = 1;
        };

        const darkModeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        isDarkMode = darkModeMediaQuery.matches;
        currentSrc = isDarkMode ? darkSrc : lightSrc;
        opacity = 1;

        darkModeMediaQuery.addEventListener("change", handleDarkModeChange);

        return () => {
            darkModeMediaQuery.removeEventListener("change", handleDarkModeChange);
        };
    });

    $: currentSrc = isDarkMode ? darkSrc : lightSrc;
</script>

<img
    src={currentSrc}
    {alt}
    class={className}
    style={`opacity: ${opacity}`}
/>