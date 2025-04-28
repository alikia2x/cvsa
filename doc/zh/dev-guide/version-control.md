# 版本控制

CVSA 是一个 monorepo 代码库，我们为每个包控制独立的版本号。在 git 中，我们使用 `包名/版本` 的格式创建标签，例如 `crawler/1.0.31`。其中的版本语法上符合 [SemVer](https://semver.org/)，但对于不同的包，语义上与 SemVer 可能不完全相同。

### backend

backend 包严格遵守 [SemVer](https://semver.org/) 规范，用于提供对外可访问的 API。

### frontend & crawler

crawler 和 frontend 包中，message 以下列字符开头的 commit 将会增加 PATCH 版本号：

* `update:`
* `fix:`
* `improve:`

message 以下列字符开头的 commit 将会增加 MINIOR 版本号：

* `add:`
* `feat:`

对于 message 以 `ref:` 开头的 commit，通常增加 PATCH 版本号。如果重构大到一定程度，增加 MAJOR 版本号。

message 以 `test:`, `merge:`, `debug:` 头的 commit，不会更改版本号。

### 根目录 package.json

根目录的 package.json 版本号 MAJOR.MINIOR.PATCH 中每个部分的值为上述三个包的对应部分之和，以反映所有包的版本变更。但需要注意，该版本不具备特别明确的语义，仅用于保证该字段能正确反映整个代码库的版本变化。
