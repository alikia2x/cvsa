# 机器学习

中V档案馆的自动化工作流高度依赖机器学习进行信息提取和分类。

我们目前使用的机器学习系统有：

#### Filter (代号 Akari）

位于项目根目录下的 `/ml/filter/`，它是一个分类模型，将来自哔哩哔哩的视频分为以下类别：

* 0：与中文歌声合成无关
* 1：中文歌声合成原创曲
* 2：中文歌声合成的翻唱/Remix歌曲

它接收三个通道的纯文本：视频的标题、简介和标签，使用一个修改后的[model2vec](https://github.com/MinishLab/model2vec)模型（从[jina-embedding-v3](https://huggingface.co/jinaai/jina-embeddings-v3)）从三个通道的文本分别产生1024维的嵌入向量作为表征，通过可学习的通道权重进行调整后送入一个隐藏层维度1296的单层全连接网络，最终连接到一个三分类器作为输出。我们使用了一个自定义的损失函数`AdaptiveRecallLoss`，以优化歌声合成作品的 recall（即使得第 0 类的 precision 尽可能高）。



此外，我们还有一些尚未投入生产的实验性工作：

#### Predictor

位于项目根目录下的 `/ml/pred/`，它预测视频的未来播放量。这是一个回归模型，它将视频的历史播放量趋势、其他上下文信息（例如当前时间）和要预测的未来时间增量作为特征输入，并输出视频播放量从“现在”到指定未来时间点的增量。

#### 歌词对齐

位于项目根目录下的 `/ml/lab/`，它分别使用 [MMS wav2vec](https://huggingface.co/docs/transformers/en/model_doc/mms) 和 [Whisper](https://github.com/openai/whisper) 模型进行音素级和行级对齐。这项工作的最初目的是驱动我们另一个项目 [AquaVox](https://github.com/alikia2x/aquavox) 中的实时歌词功能。
