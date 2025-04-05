# Scope of Inclusion

CVSA contains many aspects of Chinese Vocal Synthesis, including songs, albums, artists (publisher, manipulators, arranger, etc), singers and voice engines / voicebanks.

For a **song**, it must meet the following two conditions to be included in CVSA:

### At Leats One Line of Chinese / Chinese Virtual Singer

The lyrics of the song must contain at least one line in Chinese. Otherwise, if the lyrics of the song do not contain Chinese, it will only be included in the CVSA only if a Chinese virtual singer has been used.

We define a **Chinese virtual singer** as follows:

1. The singer primarily uses Chinese voicebank (i.e. the most widely used voickbank for the singer is Chinese)
2. The singer is operated by a company, organization, individual or group located in Mainland China, Hong Kong, Macau or\
   Taiwan.

### Using Vocal Synthesizer

To be included in CVSA, at least one line of the song must be produced by a Vocal Synthesizer (including harmony vocals).

We define a vocal synthesizer as a software or system that generates synthesized singing voices by algorithmically modeling vocal characteristics and producing audio from input parameters such as lyrics, pitch, and dynamics, encompassing both waveform-concatenation-based (e.g., VOCALOID 1\~5, UTAU) and AI-based (e.g., Synthesizer V, ACE Studio) approaches, **but excluding voice conversion tools that solely alter the timbre of pre-existing recordings** (e.g.,[so-vits svc](https://github.com/svc-develop-team/so-vits-svc)).



In addition, the songs must be featured in a video that is categorized under the VOCALOID·UTAU (ID 30) category in [Bilibili](https://en.wikipedia.org/wiki/Bilibili) in order to be observed by our [automation program](../architecure/overview.md#crawler). We welcome editors to manually add songs that have not been uploaded to bilibili / categorized under this category.

#### NEWS

Recently, Bilibili seems to be offlining the sub-category. This means the VOCALOID·UTAU category can no longer be entered from the frontend, and producers can no longer upload videos to this category (instead, they can only choose the parent category "Music").

According to our experiments, Bilibili still retains the code logic of sub-categories in the backend, and newly published songs may still be in the VOCALOID·UTAU sub-category, and the related APIs can still work normally. However, there are [reports](https://www.bilibili.com/opus/1041223385394184199) that some of the new songs have been placed under\
the "Music General" sub-category.

We are still waiting for Bilibili's follow-up actions, and in the future, we may adjust the scope of our automated program's crawling.
