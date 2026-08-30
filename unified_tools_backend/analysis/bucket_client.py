import os
import uuid
import logging
from threading import Lock
import requests

logger = logging.getLogger(__name__)


class BucketClient:

    _trace_to_artifact = {}
    _lock = Lock()

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

    def get_artifact(self, trace_id: str):
        """
        Fetch an artifact from Bucket by trace_id or artifact_id.

        Returns:
            dict | None:
                Stored artifact when found.
                None when Bucket reports 404 or artifact is not found.
        """
        if not trace_id:
            return None

        # Build list of lookup IDs: mapped artifact_id first, then trace_id
        with self._lock:
            mapped_artifact_id = self._trace_to_artifact.get(trace_id)

        candidates = []
        if mapped_artifact_id:
            candidates.append(mapped_artifact_id)
        if trace_id not in candidates:
            candidates.append(trace_id)

        for candidate_id in candidates:
            try:
                response = requests.get(
                    f"{self.base_url}/bucket/artifact/{candidate_id}",
                    timeout=15
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        "Successfully retrieved artifact for id %s (trace_id: %s)",
                        candidate_id,
                        trace_id
                    )
                    return data
                elif response.status_code == 404:
                    continue
                else:
                    response.raise_for_status()

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                logger.warning(
                    "Unable to fetch artifact from Bucket for id %s: %s",
                    candidate_id,
                    exc
                )
            except Exception as exc:
                logger.warning(
                    "Unable to fetch artifact from Bucket for id %s: %s",
                    candidate_id,
                    exc
                )

        logger.info(
            "Artifact with trace_id %s not found in Bucket.",
            trace_id
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
        artifact_id = str(uuid.uuid4())
        trace_id = canonical_intelligence.get("trace_id")

        bucket_payload = {

            "artifact_id": artifact_id,

            "trace_id":
                trace_id,

            "timestamp_utc":
                canonical_intelligence.get("timestamp"),

            "schema_version":
                canonical_intelligence.get("schema_version"),

            "source_module_id":
                "samachar",

            "artifact_type":
                "canonical_intelligence",

            "parent_hash":
                parent_hash,

            "payload":
                canonical_intelligence,
        }

        # Maintain trace_id -> artifact_id reference
        if trace_id:
            with self._lock:
                self._trace_to_artifact[trace_id] = artifact_id

        logger.info(
            "Storing artifact in Bucket. parent_hash=%s artifact_id=%s trace_id=%s",
            parent_hash,
            artifact_id,
            trace_id
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

            resp_artifact_id = resp_json.get("artifact_id")
            if resp_artifact_id and trace_id:
                with self._lock:
                    self._trace_to_artifact[trace_id] = resp_artifact_id

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