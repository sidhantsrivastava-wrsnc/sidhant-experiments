"""Video Effects Temporal worker.

Usage:
    python -m video_effects.worker
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from video_effects.activities import ALL_VIDEO_EFFECTS_ACTIVITIES
from video_effects.activities.creative import design_style
from video_effects.config import settings
from video_effects.creative_workflow import CreativeDesignerWorkflow
from video_effects.infographic_workflow import InfographicGeneratorWorkflow
from video_effects.workflow import VideoEffectsWorkflow


async def get_temporal_client() -> Client:
    """Connect to the Temporal server."""
    return await Client.connect(
        settings.TEMPORAL_ENDPOINT,
        namespace=settings.TEMPORAL_NAMESPACE,
        data_converter=pydantic_data_converter,
    )


async def main():
    client = await get_temporal_client()

    with ThreadPoolExecutor(max_workers=4) as executor:
        worker = Worker(
            client,
            task_queue=settings.TASK_QUEUE,
            workflows=[VideoEffectsWorkflow, CreativeDesignerWorkflow, InfographicGeneratorWorkflow],
            activities=[*ALL_VIDEO_EFFECTS_ACTIVITIES, design_style],
            activity_executor=executor,
        )

        print(f"Starting Video Effects worker...")
        print(f"  Task Queue: {settings.TASK_QUEUE}")
        print(f"  Namespace: {settings.TEMPORAL_NAMESPACE}")
        print(f"  Endpoint: {settings.TEMPORAL_ENDPOINT}")
        print(f"  Activities: {len(ALL_VIDEO_EFFECTS_ACTIVITIES)}")

        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
