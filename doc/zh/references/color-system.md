# 颜色系统

### 对比度

我们使用 APCA (Advanced Perceptual Contrast Algorithm, 先进感知对比算法) 测量前景色与背景色的对比度，并以相较于参考值降低 15% 为最低可容忍的对比度，以平衡无障碍与个性。

在大多数情况下，参考值是字号 16px，字重 400 下 APCA 速查表中的对比度参考值，即 90 Lc。因此，降低 15% 意味着在大多数情况下，设计选用的颜色组合的对比度绝对值应达到至少 76.5 Lc。

### 背景色

我们使用 ![十六进制色号为18181B的纯色图片](../.gitbook/assets/#18181b.svg) #18181B, 即 rgb(24 24 27) 作为深色模式下的背景色，而使用纯白色 ![](../.gitbook/assets/#FFFFFF.svg) #ffffff，即 rgb(255 255 255) 作为浅色模式下的背景色。

### 基色

用于生成色卡的基色为 ![](../.gitbook/assets/#EE00000.svg) #ee0000, 即 rgb(238 0 0)。它将会作为 _Seed Color_ 在 [Material Theme Builder](https://material-foundation.github.io/material-theme-builder/) 中生成色卡。

### 主题色

对于深色模式，我们使用 ![](../.gitbook/assets/#FFC1B8.svg) #ffc1b8, 即 rgb(255 193 184) 作为主题色。该颜色在色卡的 **Primary Color** (#ffb4a8) 上做出了略微调整，使其与背景的对比度达到 Lc 76.5。

对于浅色模式，我们使用 ![](../.gitbook/assets/#904B40.svg) #904b40, 即 rgb(144 75 64) 作为主题色。

