# Database Structure

CVSA uses [PostgreSQL](https://www.postgresql.org/) as our database.

All public data of CVSA (excluding users' personal data) is stored in a database named `cvsa_main`, which contains the following tables:

* songs: stores the main information of songs
* bili\_user: stores snapshots of Bilibili user information
* all\_data: metadata of all videos in [category 30](../../about/scope-of-inclusion.md#category-30).
* labelling\_result: Contains label of videos in `all_data`tagged by our [AI system](../artificial-intelligence.md#the-filter).
* video\_snapshot: Statistical data of videos that are fetched regularly (e.g., number of views, etc.), we call this fetch process as "snapshot".
* snapshot\_schedule: The scheduling information for video snapshots.

