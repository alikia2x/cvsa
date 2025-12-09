"""
Database connection and operations for ML training service
"""

import os
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncpg
import logging
from config_loader import config_loader
from dotenv import load_dotenv

load_dotenv() 

logger = logging.getLogger(__name__)

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
                SELECT data_checksum, vec_2048, vec_1536, vec_1024, created_at
                FROM internal.embeddings
                WHERE model_name = $1 AND data_checksum = ANY($2::text[])
            """
            rows = await conn.fetch(query, model_name, checksums)

            result = {}
            for row in rows:
                checksum = row["data_checksum"]
                
                # Convert vector strings to lists if they exist
                vec_2048 = self._parse_vector_string(row["vec_2048"]) if row["vec_2048"] else None
                vec_1536 = self._parse_vector_string(row["vec_1536"]) if row["vec_1536"] else None
                vec_1024 = self._parse_vector_string(row["vec_1024"]) if row["vec_1024"] else None
                
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
            if vector_str.startswith('[') and vector_str.endswith(']'):
                vector_str = vector_str[1:-1]
            
            return [float(x.strip()) for x in vector_str.split(',') if x.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse vector string '{vector_str}': {e}")
            return []

    async def insert_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> None:
        """Batch insert embeddings into database"""
        if not embeddings_data:
            return

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for data in embeddings_data:
                    # Determine which vector column to use based on dimensions
                    vec_column = f"vec_{data['dimensions']}"
                    
                    # Convert vector list to string format for PostgreSQL
                    vector_str = "[" + ",".join(map(str, data["vector"])) + "]"

                    query = f"""
                        INSERT INTO internal.embeddings
                        (model_name, data_checksum, {vec_column}, created_at)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (data_checksum) DO NOTHING
                    """

                    await conn.execute(
                        query,
                        data["model_name"],
                        data["checksum"],
                        vector_str,
                        datetime.now(),
                    )

    async def get_final_dataset(
        self, aid_list: List[int], model_name: str
    ) -> List[Dict[str, Any]]:
        """Get final dataset with embeddings and labels"""
        if not aid_list:
            return []

        # Get video metadata
        metadata = await self.get_video_metadata(aid_list)

        # Get user labels (latest per user)
        labels = await self.get_user_labels(aid_list)

        # Prepare text data for embedding
        text_data = []
        aid_to_text = {}

        for aid in aid_list:
            if aid in metadata:
                # Combine title, description, and tags for embedding
                text_parts = [
                    metadata[aid]["title"],
                    metadata[aid]["description"],
                    metadata[aid]["tags"],
                ]
                combined_text = " ".join(filter(None, text_parts))

                # Create checksum for deduplication
                checksum = hashlib.md5(combined_text.encode("utf-8")).hexdigest()

                text_data.append(
                    {"aid": aid, "text": combined_text, "checksum": checksum}
                )
                aid_to_text[checksum] = aid

        # Get checksums = [ existing embeddings
        checks = [item["checksum"] for item in text_data]
        existing_embeddings = await self.get_existing_embeddings(checks)

        # ums, model_name Prepare final dataset
        dataset = []

        for item in text_data:
            aid = item["aid"]
            checksum = item["checksum"]

            # Get embedding vector
            embedding_vector = None
            if checksum in existing_embeddings:
                # Use existing embedding
                emb_data = existing_embeddings[checksum]
                if emb_data["vec_1536"]:
                    embedding_vector = emb_data["vec_1536"]
                elif emb_data["vec_2048"]:
                    embedding_vector = emb_data["vec_2048"]
                elif emb_data["vec_1024"]:
                    embedding_vector = emb_data["vec_1024"]

            # Get labels for this aid
            aid_labels = labels.get(aid, [])

            # Determine final label using consensus (majority vote)
            if aid_labels:
                positive_votes = sum(1 for lbl in aid_labels if lbl["label"])
                final_label = positive_votes > len(aid_labels) / 2
            else:
                final_label = None  # No labels available

            # Check for inconsistent labels
            inconsistent = len(aid_labels) > 1 and (
                sum(1 for lbl in aid_labels if lbl["label"]) != 0
                and sum(1 for lbl in aid_labels if lbl["label"]) != len(aid_labels)
            )

            if embedding_vector and final_label is not None:
                dataset.append(
                    {
                        "aid": aid,
                        "embedding": embedding_vector,
                        "label": final_label,
                        "metadata": metadata.get(aid, {}),
                        "user_labels": aid_labels,
                        "inconsistent": inconsistent,
                        "text_checksum": checksum,
                    }
                )

        return dataset


# Global database manager instance
db_manager = DatabaseManager()
