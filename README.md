# 中V档案馆 - Chinese Vocal Synthesis Archive

「中V档案馆」是一个旨在收录与展示「中文歌声合成作品」及有关信息的网站。

## 创建背景与关联工作

纵观整个互联网，对于「中文歌声合成」或「中文虚拟歌手」（常简称为中V或VC）相关信息进行较为系统、全面地整理收集的主要有以下几个网站：

- [萌娘百科](https://zh.moegirl.org.cn/):
  收录了大量中V歌曲及歌姬的信息，呈现形式为传统维基（基于[MediaWiki](https://www.mediawiki.org/)）。
- [VCPedia](https://vcpedia.cn/):
  由原萌娘百科中文歌声合成编辑团队的部分成员搭建，专属于中文歌声合成相关内容的信息集成站点[^1]，呈现形式为传统维基（基于[MediaWiki](https://www.mediawiki.org/)）。
- [VocaDB](https://vocadb.net/): 一个围绕 Vocaloid、UTAU 和其他歌声合成器的协作数据库，其中包含艺术家、唱片、PV
  等[^2]，其中包含大量中文歌声合成作品。
- [天钿Daily](https://tdd.bunnyxt.com/)：一个VC相关数据交流与分享的网站。致力于VC相关数据交流，定期抓取VC相关数据，选取有意义的纬度展示。[^3]

上述网站中，或多或少存在一些不足，例如：

- 萌娘百科、VCPedia受限于传统维基，绝大多数内容依赖人工编辑。
- VocaDB基于结构化数据库构建，由此可以依赖程序生成一些信息，但**条目收录**仍然完全依赖人工完成。
- VocaDB主要专注于元数据展示，少有关于歌曲、作者等的描述性的文字，也缺乏描述性的背景信息。
- 天钿Daily只展示歌曲的统计数据及历史趋势，没有关于歌曲其它信息的收集。

因此，**中V档案馆**吸取前人经验，克服上述网站的不足，希望做到：

- 歌曲收录（指发现歌曲并创建条目）的完全自动化
- 歌曲元信息提取的高度自动化
- 歌曲统计数据收集的完全自动化
- 在程序辅助的同时欢迎并鼓励贡献者参与编辑（主要为描述性内容）或纠错
- 在适当的许可声明下，引用来自上述源的数据，使内容更加全面、丰富。

## 技术架构

参见[CVSA文档](https://cvsa.gitbook.io/)。

## 开放许可

受本文以[CC BY-NC-SA 4.0协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)提供。

### 数据库

中V档案馆使用[PostgreSQL](https://postgresql.org)作为数据库，我们承诺定期导出数据库转储 (dump)
文件并公开，其内容遵从以下协议或条款：

- 数据库中的事实性数据，根据适用法律，不构成受版权保护的内容。中V档案馆放弃一切可能的权利（[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)）。
- 对于数据库中有原创性的内容（如贡献者编辑的描述性内容），如无例外，以[CC BY 4.0协议](https://creativecommons.org/licenses/by/4.0/)提供。
- 对于引用、摘编或改编自萌娘百科、VCPedia的内容，以与原始协议(CC BY-NC-SA 3.0
  CN)兼容的协议[CC BY-NC-SA 4.0协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)提供，并注明原始协议 。
  > 根据原始协议第四条第2项内容，CC BY-NC-SA 4.0协议为与原始协议具有相同授权要素的后续版本（“可适用的协议”）。
- 中V档案馆文档使用[CC BY 4.0协议](https://creativecommons.org/licenses/by/4.0/)。

### 软件代码

用于构建中V档案馆的软件代码在[AGPL 3.0](https://www.gnu.org/licenses/agpl-3.0.html)许可证下公开，参见[LICENSE](./LICENSE)

[^1]: 引用自[VCPedia](https://vcpedia.cn/%E9%A6%96%E9%A1%B5)，于[知识共享 署名-非商业性使用-相同方式共享 3.0中国大陆 (CC BY-NC-SA 3.0 CN) 许可协议](https://creativecommons.org/licenses/by-nc-sa/3.0/cn/)下提供。

[^2]: 翻译自[VocaDB](https://vocadb.net/)，于[CC BY 4.0协议](https://creativecommons.org/licenses/by/4.0/)下提供。

[^3]: 引用自[关于 - 天钿Daily](https://tdd.bunnyxt.com/about)
