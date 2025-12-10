"""
Database connection and operations for ML training service
"""

from collections import defaultdict
import os
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncpg
from config_loader import config_loader
from dotenv import load_dotenv
from logger_config import get_logger

load_dotenv()

logger = get_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)

            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    @property
    def is_connected(self) -> bool:
        """Check if database connection pool is initialized"""
        return self.pool is not None

    async def get_embedding_models(self):
        """Get available embedding models from config"""
        return config_loader.get_embedding_models()

    async def get_video_metadata(
        self, aid_list: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Get video metadata for given AIDs"""
        if not aid_list:
            return {}

        async with self.pool.acquire() as conn:
            query = """
                SELECT aid, title, description, tags 
                FROM bilibili_metadata 
                WHERE aid = ANY($1::bigint[])
            """
            rows = await conn.fetch(query, aid_list)

            result = {}
            for row in rows:
                result[int(row["aid"])] = {
                    "aid": int(row["aid"]),
                    "title": row["title"] or "",
                    "description": row["description"] or "",
                    "tags": row["tags"] or "",
                }

            return result

    async def get_user_labels(
        self, aid_list: List[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Get user labels for given AIDs, only the latest label per user"""
        if not aid_list:
            return {}

        async with self.pool.acquire() as conn:
            query = """
                WITH latest_labels AS (
                    SELECT DISTINCT ON (aid, "user") 
                        aid, "user", label, created_at
                    FROM internal.video_type_label 
                    WHERE aid = ANY($1::bigint[])
                    ORDER BY aid, "user", created_at DESC
                )
                SELECT aid, "user", label, created_at
                FROM latest_labels
                ORDER BY aid, "user"
            """
            rows = await conn.fetch(query, aid_list)

            result = {}
            for row in rows:
                aid = int(row["aid"])
                if aid not in result:
                    result[aid] = []

                result[aid].append(
                    {
                        "user": row["user"],
                        "label": bool(row["label"]),
                        "created_at": row["created_at"].isoformat(),
                    }
                )

            return result

    async def get_existing_embeddings(
        self, checksums: List[str], model_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get existing embeddings for given checksums and model"""
        if not checksums:
            return {}

        async with self.pool.acquire() as conn:
            query = """
                SELECT data_checksum, dimensions, vec_2048, vec_1536, vec_1024, created_at
                FROM internal.embeddings
                WHERE model_name = $1 AND data_checksum = ANY($2::text[])
            """
            rows = await conn.fetch(query, model_name, checksums)

            result = {}
            for row in rows:
                checksum = row["data_checksum"]

                # Convert vector strings to lists if they exist
                vec_2048 = (
                    self._parse_vector_string(row["vec_2048"])
                    if row["vec_2048"]
                    else None
                )
                vec_1536 = (
                    self._parse_vector_string(row["vec_1536"])
                    if row["vec_1536"]
                    else None
                )
                vec_1024 = (
                    self._parse_vector_string(row["vec_1024"])
                    if row["vec_1024"]
                    else None
                )

                result[checksum] = {
                    "checksum": checksum,
                    "vec_2048": vec_2048,
                    "vec_1536": vec_1536,
                    "vec_1024": vec_1024,
                    "created_at": row["created_at"].isoformat(),
                }

            return result

    def _parse_vector_string(self, vector_str: str) -> List[float]:
        """Parse vector string format '[1.0,2.0,3.0]' back to list"""
        if not vector_str:
            return []

        try:
            # Remove brackets and split by comma
            vector_str = vector_str.strip()
            if vector_str.startswith("[") and vector_str.endswith("]"):
                vector_str = vector_str[1:-1]

            return [float(x.strip()) for x in vector_str.split(",") if x.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse vector string '{vector_str}': {e}")
            return []

    async def insert_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> None:
        """Batch insert embeddings into database (Optimized)"""
        if not embeddings_data:
            return

        batches = defaultdict(list)
        now = datetime.now()

        for data in embeddings_data:
            vector_str = str(data["vector"])
            # "[" + ",".join(map(str, data["vector"])) + "]"

            dim = data["dimensions"]

            batches[dim].append(
                (data["model_name"], dim, data["checksum"], vector_str, now)
            )

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for dim, values in batches.items():
                    vec_column = f"vec_{dim}"

                    query = f"""
                        INSERT INTO internal.embeddings
                        (model_name, dimensions, data_checksum, {vec_column}, created_at)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (model_name, dimensions, data_checksum) DO NOTHING
                    """

                    await conn.executemany(query, values)

    # Sampling Methods
    async def get_all_aids(self) -> List[int]:
        """Get all available AIDs from labeled data (internal.video_type_label)"""
        async with self.pool.acquire() as conn:
            query = "SELECT DISTINCT aid FROM internal.video_type_label WHERE aid IS NOT NULL"
            rows = await conn.fetch(query)
            return [int(row["aid"]) for row in rows]
        
    async def get_all_aids_count(self) -> List[int]:
        """Get all available AIDs from labeled data (internal.video_type_label)"""
        async with self.pool.acquire() as conn:
            query = "SELECT COUNT(DISTINCT aid) FROM internal.video_type_label WHERE aid IS NOT NULL"
            rows = await conn.fetch(query)
            return rows[0]["count"]

    async def get_aids_by_strategy(
        self, strategy: str, limit: Optional[int] = None
    ) -> List[int]:
        """Get AIDs based on sampling strategy"""
        if strategy == "all":
            return await self.get_all_aids()
        elif strategy == "random":
            return await self.get_random_aids(limit or 1000)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    async def get_random_aids(
        self, limit: int
    ) -> List[int]:
        """Get random AIDs from labeled data only"""
        async with self.pool.acquire() as conn:
            query = "SELECT aid FROM internal.video_type_label ORDER BY RANDOM() LIMIT $1"
            rows = await conn.fetch(query, limit)
            aids = [int(row["aid"]) for row in rows]
            # deduplication
            return list(set(aids))

    async def get_sampling_stats(self) -> Dict[str, Any]:
        """Get statistics about available labeled data for sampling"""
        async with self.pool.acquire() as conn:
            # Total labeled videos
            total_labeled_query = (
                "SELECT COUNT(DISTINCT aid) as count FROM internal.video_type_label"
            )
            total_labeled_result = await conn.fetchrow(total_labeled_query)
            total_labeled_videos = total_labeled_result["count"]

            # Positive and negative labels
            positive_query = "SELECT COUNT(DISTINCT aid) as count FROM internal.video_type_label WHERE label = true"
            negative_query = "SELECT COUNT(DISTINCT aid) as count FROM internal.video_type_label WHERE label = false"

            positive_result = await conn.fetchrow(positive_query)
            negative_result = await conn.fetchrow(negative_query)

            positive_labels = positive_result["count"]
            negative_labels = negative_result["count"]

            return {
                "total_labeled_videos": total_labeled_videos,
                "positive_labels": positive_labels,
                "negative_labels": negative_labels,
            }


# Global database manager instance
db_manager = DatabaseManager()
