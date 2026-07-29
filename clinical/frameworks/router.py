from __future__ import annotations

import asyncio
import time

from app_logging.logger import get_logger
from clinical.frameworks.base import BaseFramework
from clinical.frameworks.models import FrameworkContext, FrameworkResult

_log = get_logger(__name__)


class ParallelFrameworkRouter:
    """Fans out to all registered frameworks concurrently and gathers results.

    Builds a ``FrameworkContext`` from pipeline state and passes it to
    each framework's ``analyze()`` method.  Each framework produces its
    own analysis through its unique theoretical lens.
    """

    async def route(
        self,
        frameworks: list[BaseFramework],
        context: FrameworkContext,
    ) -> dict[str, FrameworkResult]:
        """Run all frameworks in parallel.

        Args:
            frameworks: List of framework plugins to invoke.
            context:    Typed context containing all case data and LLM service.

        Returns:
            Dict mapping ``framework.name -> FrameworkResult``.
        """
        if not frameworks:
            _log.warning("framework_router.no_frameworks")
            return {}

        async def _run_one(fw: BaseFramework) -> tuple[str, FrameworkResult]:
            t0 = time.perf_counter()
            try:
                result = await fw.analyze(context)
                elapsed = (time.perf_counter() - t0) * 1000
                object.__setattr__(result, "analysis_ms", round(elapsed, 2))
                _log.info(
                    "framework_router.framework_complete",
                    framework=fw.name,
                    ms=round(elapsed, 2),
                )
                return fw.name, result
            except Exception as exc:
                _log.error(
                    "framework_router.framework_failed",
                    framework=fw.name,
                    error=str(exc),
                )
                elapsed = (time.perf_counter() - t0) * 1000
                return fw.name, FrameworkResult(
                    framework_name=fw.name,
                    framework_description=fw.description,
                    conceptualization="Analysis failed.",
                    confidence=0.0,
                    analysis_ms=round(elapsed, 2),
                )

        results = await asyncio.gather(*[_run_one(fw) for fw in frameworks])
        return dict(results)
