"use client";

import TitleLight from "@/public/icons/标题-浅色.svg";
import TitleDark from "@/public/icons/标题-深色.svg";
import LogoMobileLight from "@/public/icons/TitleBar Mobile Light.svg";
import LogoMobileDark from "@/public/icons/TitleBar Mobile Dark.svg";
import DarkModeImage from "@/components/utils/DarkModeImage";
import React, { useState } from "react";
import { NavigationDrawer } from "@/components/ui/NavigatinDrawer";
import { Portal } from "@/components/utils/Portal";
import { RegisterIcon } from "@/components/icons/RegisterIcon";
import { SearchBox } from "@/components/ui/SearchBox";
import { MenuIcon } from "@/components/icons/MenuIcon";
import { SearchIcon } from "@/components/icons/SearchIcon";
import { InfoIcon } from "@/components/icons/InfoIcon";
import { HomeIcon } from "@/components/icons/HomeIcon";
import { TextButton } from "@/components/ui/Buttons/TextButton";
import { Link } from "@/i18n/navigation";

export const HeaderDestop = () => {
	return (
		<div className="hidden md:flex relative top-0 left-0 w-full h-28 z-20 justify-between">
			<div className="w-[305px] xl:ml-8 inline-flex items-center">
				<a href="/">
					<DarkModeImage
						lightSrc={TitleLight}
						darkSrc={TitleDark}
						alt="logo"
						className="w-[305px] h-24 inline-block max-w-[15rem] lg:max-w-[305px]"
					/>
				</a>
			</div>

			<SearchBox />

			<div
				className="inline-flex relative gap-6 h-full lg:right-12
    				text-xl font-medium items-center w-[15rem] min-w-[8rem] mr-4 lg:mr-0 lg:w-[305px] justify-end"
			>
				<a href="/signup">注册</a>
				<a href="/about">关于</a>
			</div>
		</div>
	);
};

export const HeaderMobile = () => {
	const [showDrawer, setShowDrawer] = useState(false);
	const [showsearchBox, setShowsearchBox] = useState(false);

	return (
		<>
			<Portal>
				<NavigationDrawer show={showDrawer} onClose={() => setShowDrawer(false)}>
					<div className="flex flex-col w-full gap-2">
						<div className="w-full h-14 flex items-center px-4 mt-3 pl-6">
							<DarkModeImage
								lightSrc={LogoMobileLight}
								darkSrc={LogoMobileDark}
								alt="Logo"
								className="w-30 h-10"
							/>
						</div>

						<Link href="/">
							<TextButton className="w-full h-14 flex px-4 justify-start" size="m">
								<div className="flex items-center">
									<HomeIcon className="text-2xl pr-4" />
									<span>首页</span>
								</div>
							</TextButton>
						</Link>
						<Link href="/about">
							<TextButton className="w-full h-14 flex px-4 justify-start" size="m">
								<div className="flex items-center">
									<InfoIcon className="text-2xl pr-4" />
									<span>关于</span>
								</div>
							</TextButton>
						</Link>

						<Link href="/signup">
							<TextButton className="w-full h-14 flex px-4 justify-start" size="m">
								<div className="flex items-center">
									<RegisterIcon className="text-2xl pr-4" />
									<span>注册</span>
								</div>
							</TextButton>
						</Link>
					</div>
				</NavigationDrawer>
			</Portal>
			<div className="md:hidden relative top-0 left-0 w-full h-16 z-20">
				{!showsearchBox && (
					<button
						className="inline-flex absolute left-0 ml-4 h-full items-center dark:text-white  text-2xl"
						onClick={() => setShowDrawer(true)}
					>
						<MenuIcon />
					</button>
				)}
				{!showsearchBox && (
					<div className="absolute left-1/2 -translate-x-1/2 -translate-y-0.5 inline-flex h-full items-center">
						<Link href="/">
							<DarkModeImage
								lightSrc={LogoMobileLight}
								darkSrc={LogoMobileDark}
								alt="Logo"
								className="w-24 h-8 translate-y-[2px]"
							/>
						</Link>
					</div>
				)}
				{showsearchBox && <SearchBox close={() => setShowsearchBox(false)} />}
				{!showsearchBox && (
					<button
						className="inline-flex absolute right-0 h-full items-center mr-4"
						onClick={() => setShowsearchBox(!showsearchBox)}
					>
						<SearchIcon className="text-[1.625rem]" />
					</button>
				)}
			</div>
		</>
	);
};

export const Header = () => {
	return (
		<>
			<HeaderDestop />
			<HeaderMobile />
		</>
	);
};
