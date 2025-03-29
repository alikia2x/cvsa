<script lang="ts">
    import logoMobileDark from "@assets/TitleBar Mobile Dark.svg";
    import logoMobileLight from "@assets/TitleBar Mobile Light.svg";
    import SearchBox from "./SearchBox.svelte";
    import SearchIcon from "./SearchIcon.svelte";
    import MenuIcon from "./MenuIcon.svelte";
    import DarkModeImage from "./DarkModeImage.svelte";
    import CloseIcon from "./CloseIcon.svelte";

    let showSearchBox = false;
</script>

<div class="md:hidden fixed top-0 left-0 w-full h-16 bg-white/80 dark:bg-zinc-800/70 backdrop-blur-lg z-100">
    {#if !showSearchBox}
        <div class="inline-block ml-4 mt-4 text-white">
            <MenuIcon />
        </div>
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
        <SearchBox autoFocus={true} />
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
