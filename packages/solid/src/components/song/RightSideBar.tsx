import { Component } from "solid-js";
import { ExtendedFAB } from "@m3-components/solid";
import { EditIcon } from "~/components/icons";
import { TabSwitcher } from "~/components/song/TabSwitcher";

export const RightSideBar: Component = () => {
	return (
		<>
			<div class="w-48 self-center 2xl:self-end flex justify-end mb-6">
				<ExtendedFAB position="unset" size="small" elevation={false} text="编辑" color="primary">
					<EditIcon />
				</ExtendedFAB>
			</div>
			<TabSwitcher />
		</>
	);
};
