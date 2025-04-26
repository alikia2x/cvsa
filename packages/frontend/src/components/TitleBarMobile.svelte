<script lang="ts">
    import logoMobileDark from "@assets/TitleBar Mobile Dark.svg";
    import logoMobileLight from "@assets/TitleBar Mobile Light.svg";
    import SearchBox from "./SearchBox.svelte";
    import SearchIcon from "./SearchIcon.svelte";
    import MenuIcon from "./MenuIcon.svelte";
    import DarkModeImage from "./DarkModeImage.svelte";
    import CloseIcon from "./CloseIcon.svelte";

    let searchBox: SearchBox | null = null;
    let showSearchBox = false;

    $: if (showSearchBox && searchBox) {
        searchBox.changeFocusState(true);
    }
</script>

<div class="md:hidden relative top-0 left-0 w-full h-16 bg-white/80 dark:bg-zinc-800/70 backdrop-blur-lg z-50">
    {#if !showSearchBox}
        <button class="inline-block ml-4 mt-4 dark:text-white">
            <MenuIcon />
        </button>
        <div class="ml-8 inline-flex h-full items-center">
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
        <SearchBox bind:this={searchBox} />
    {/if}
    <button
        class="inline-flex absolute right-0 h-full items-center mr-4"
        onclick={() => (showSearchBox = !showSearchBox)}
    >
        {#if showSearchBox}
            <CloseIcon />
        {:else}
            <SearchIcon />
        {/if}
    </button>
</div>
