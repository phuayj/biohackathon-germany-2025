from __future__ import annotations


class Network:
    def __init__(
        self,
        height: str,
        width: str,
        bgcolor: str,
        font_color: str,
        directed: bool = ...,
        notebook: bool = ...,
        select_menu: bool = ...,
        filter_menu: bool = ...,
    ) -> None: ...

    def set_options(self, options: str) -> None: ...

    def add_node(
        self,
        n_id: str,
        label: str | None = ...,
        title: str | None = ...,
        color: object | None = ...,
        shape: str | None = ...,
        size: float | int | None = ...,
        borderWidth: float | int | None = ...,
        borderWidthSelected: float | int | None = ...,
    ) -> None: ...

    def add_edge(
        self,
        source: str,
        to: str,
        label: str | None = ...,
        title: str | None = ...,
        color: object | None = ...,
        width: float | int | None = ...,
        dashes: bool | None = ...,
        arrows: str | None = ...,
    ) -> None: ...

    def generate_html(self, name: str | None = ...) -> str: ...

