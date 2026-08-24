# import os
# import uuid
# import logging
# import requests

# logger = logging.getLogger(__name__)


# class BucketClient:

#     def __init__(self):

#         self.base_url = os.getenv("BUCKET_URL")
#         self._last_response_hash = None

#         if not self.base_url:
#             raise RuntimeError(
#                 "BUCKET_URL is not configured."
#             )

#     def get_latest_hash(self):

#         try:

#             response = requests.get(
#                 f"{self.base_url}/bucket/latest-hash",
#                 timeout=15
#             )

#             response.raise_for_status()

#             data = response.json()

#             return data.get("last_hash")

#         except Exception as exc:

#             logger.warning(
#                 "Unable to fetch latest bucket hash: %s",
#                 exc
#             )

#             return None

#     def store_artifact(
#         self,
#         canonical_intelligence: dict
#     ):

#         parent_hash = self.get_latest_hash() or getattr(self, "_last_response_hash", None)

#         bucket_payload = {

#             "artifact_id": str(uuid.uuid4()),

#             "trace_id":
#                 canonical_intelligence["trace_id"],

#             "timestamp_utc":
#                 canonical_intelligence["timestamp"],

#             "schema_version":
#                 canonical_intelligence["schema_version"],

#             "source_module_id":
#                 "samachar",

#             "artifact_type":
#                 "canonical_intelligence",

#             "parent_hash":
#                 parent_hash,

#             "payload":
#                 canonical_intelligence,
#         }

#         response = requests.post(
#             f"{self.base_url}/bucket/artifact",
#             json=bucket_payload,
#             timeout=30
#         )

#         response.raise_for_status()

#         resp_json = response.json()

#         # Cache a returned hash from the artifact response so the next call
#         # can use it as parent when latest-hash endpoint returns null.
#         if isinstance(resp_json, dict):
#             for key in ("parent_hash", "hash", "artifact_hash", "last_hash"):
#                 if resp_json.get(key):
#                     self._last_response_hash = resp_json.get(key)
#                     break

#         logger.info(
#             "Artifact stored successfully in Bucket."
#         )

#         return resp_json


import os
import uuid
import logging
import requests

logger = logging.getLogger(__name__)


class BucketClient:

    def __init__(self):

        self.base_url = os.getenv("BUCKET_URL")
        self._last_response_hash = None

        if not self.base_url:
            raise RuntimeError(
                "BUCKET_URL is not configured."
            )

    def get_latest_hash(self):
        """
        Fetch the latest hash from Bucket.

        Returns:
            str | None:
                Latest Bucket hash when available.
                None when Bucket reports no latest hash or
                the endpoint cannot be reached.
        """

        try:

            response = requests.get(
                f"{self.base_url}/bucket/latest-hash",
                timeout=15
            )

            response.raise_for_status()

            data = response.json()

            latest_hash = data.get("last_hash")

            logger.info(
                "Bucket latest hash: %s",
                latest_hash
            )

            return latest_hash

        except Exception as exc:

            logger.warning(
                "Unable to fetch latest bucket hash: %s",
                exc
            )

            return None

    def _resolve_parent_hash(self):
        """
        Determine the parent hash for the next artifact.

        Priority:

        1. Bucket /latest-hash value, if available.
        2. Hash returned by the previous successful /artifact call.
        3. None for the first artifact.

        Important:
        The hash returned by /artifact represents the newly
        created artifact and becomes the parent_hash for the
        next artifact.
        """

        latest_hash = self.get_latest_hash()

        if latest_hash:
            logger.info(
                "Using Bucket latest hash as parent_hash: %s",
                latest_hash
            )

            return latest_hash

        if self._last_response_hash:
            logger.info(
                "Bucket latest hash is null. "
                "Using previous artifact hash as parent_hash: %s",
                self._last_response_hash
            )

            return self._last_response_hash

        logger.info(
            "Bucket latest hash is null and no previous artifact "
            "hash exists. Using parent_hash=null for first artifact."
        )

        return None

    def store_artifact(
        self,
        canonical_intelligence: dict
    ):
        """
        Store canonical intelligence in Bucket.

        Parent-hash behavior:

        First artifact:
            parent_hash = None

        Subsequent artifact:
            parent_hash = hash returned by the previous
            successful /artifact request, unless
            /latest-hash provides a valid hash.

        The hash generated by Bucket for the current artifact
        is stored locally and becomes the parent_hash for the
        next artifact when /latest-hash returns null.
        """

        parent_hash = self._resolve_parent_hash()

        bucket_payload = {

            "artifact_id": str(uuid.uuid4()),

            "trace_id":
                canonical_intelligence["trace_id"],

            "timestamp_utc":
                canonical_intelligence["timestamp"],

            "schema_version":
                canonical_intelligence["schema_version"],

            "source_module_id":
                "samachar",

            "artifact_type":
                "canonical_intelligence",

            "parent_hash":
                parent_hash,

            "payload":
                canonical_intelligence,
        }

        logger.info(
            "Storing artifact in Bucket. parent_hash=%s",
            parent_hash
        )

        response = requests.post(
            f"{self.base_url}/bucket/artifact",
            json=bucket_payload,
            timeout=30
        )

        response.raise_for_status()

        resp_json = response.json()

        # --------------------------------------------------
        # IMPORTANT:
        # Only update the cached parent hash AFTER Bucket
        # successfully stores the artifact.
        # --------------------------------------------------

        if isinstance(resp_json, dict):

            generated_hash = resp_json.get("hash")

            if generated_hash:

                self._last_response_hash = generated_hash

                logger.info(
                    "Bucket generated artifact hash: %s",
                    generated_hash
                )

            else:

                logger.warning(
                    "Bucket artifact response did not contain "
                    "a generated hash."
                )

        logger.info(
            "Artifact stored successfully in Bucket."
        )

        return resp_json