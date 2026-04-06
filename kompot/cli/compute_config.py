"""
Compute configuration for JAX, NumPy, and Dask.

This module handles GPU/CPU configuration and thread limiting for computational backends.

IMPORTANT NOTES:
1. NumPy thread limits: Set early in main() via environment variables BEFORE NumPy import.
   The _configure_thread_limits() function here is called later but only affects subsequently
   loaded modules (like Dask), not NumPy which is already initialized.

2. JAX configuration: Must be called AFTER mellon import, as mellon configures JAX on import.
   The _configure_jax() function can override mellon's settings.

3. Dask configuration: Can be set at any time via dask.config.
"""

import os
import logging

logger = logging.getLogger("kompot.cli")


def configure_compute(use_gpu: bool = False, n_threads: int = None):
    """
    Configure computational backends (JAX, NumPy, Dask) for thread control and GPU usage.

    This function must be called AFTER importing mellon, as mellon configures JAX
    to use CPU on import. This function can override that configuration.

    Parameters
    ----------
    use_gpu : bool, default=False
        If True, configure JAX to use GPU. If False, force CPU usage.
    n_threads : int, optional
        Number of threads to use. If specified, limits threads for:
        - JAX (XLA)
        - NumPy (OpenBLAS/MKL)
        - Dask

    Notes
    -----
    Thread limiting affects:
    - JAX: Set via XLA_FLAGS environment variable
    - NumPy: Set via OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS
    - Dask: Set via dask.config

    Examples
    --------
    >>> # CPU-only with 4 threads
    >>> configure_compute(use_gpu=False, n_threads=4)

    >>> # GPU with thread limiting
    >>> configure_compute(use_gpu=True, n_threads=8)
    """
    logger.info("=" * 60)
    logger.info("Configuring computational backends")
    logger.info("=" * 60)

    # Configure thread limits BEFORE JAX initialization
    if n_threads is not None:
        logger.info(f"Setting thread limit: {n_threads} threads")
        _configure_thread_limits(n_threads)
    else:
        logger.info("No thread limit specified (using system defaults)")

    # Configure JAX (must be done AFTER mellon import)
    _configure_jax(use_gpu, n_threads)

    # Configure Dask if available
    try:
        _configure_dask(n_threads)
    except ImportError:
        logger.debug("Dask not available, skipping dask configuration")

    logger.info("=" * 60)


def _configure_thread_limits(n_threads: int):
    """
    Set environment variables to limit threads for NumPy and related libraries.

    Parameters
    ----------
    n_threads : int
        Number of threads to use
    """
    n_threads_str = str(n_threads)

    # OpenMP (used by NumPy, SciPy, etc.)
    os.environ["OMP_NUM_THREADS"] = n_threads_str
    logger.debug(f"  Set OMP_NUM_THREADS={n_threads_str}")

    # Intel MKL (if NumPy is built with MKL)
    os.environ["MKL_NUM_THREADS"] = n_threads_str
    logger.debug(f"  Set MKL_NUM_THREADS={n_threads_str}")

    # OpenBLAS (if NumPy is built with OpenBLAS)
    os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
    logger.debug(f"  Set OPENBLAS_NUM_THREADS={n_threads_str}")

    # BLAS (general)
    os.environ["BLAS_NUM_THREADS"] = n_threads_str
    logger.debug(f"  Set BLAS_NUM_THREADS={n_threads_str}")

    logger.info(f"  NumPy/BLAS thread limit: {n_threads} threads")


def _configure_jax(use_gpu: bool, n_threads: int = None):
    """
    Configure JAX for GPU/CPU usage and thread limiting.

    Must be called AFTER mellon import, as mellon sets JAX to CPU mode on import.

    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU
    n_threads : int, optional
        Number of threads for CPU execution
    """
    import jax

    if use_gpu:
        # Check if GPU is available
        try:
            devices = jax.devices("gpu")
            if len(devices) > 0:
                logger.info("  JAX: GPU mode enabled")
                logger.info(f"    Available GPU devices: {len(devices)}")
                for i, device in enumerate(devices):
                    logger.info(f"      Device {i}: {device}")

                # Set default device to GPU
                # Note: mellon may have set it to CPU, we override here
                jax.config.update("jax_platform_name", "gpu")
            else:
                logger.warning(
                    "  JAX: GPU requested but no GPU devices found, falling back to CPU"
                )
                jax.config.update("jax_platform_name", "cpu")
                use_gpu = False
        except RuntimeError as e:
            logger.warning(f"  JAX: GPU not available ({e}), using CPU")
            jax.config.update("jax_platform_name", "cpu")
            use_gpu = False
    else:
        logger.info("  JAX: CPU mode (GPU disabled)")
        jax.config.update("jax_platform_name", "cpu")

    # Configure thread limits for JAX/XLA
    if not use_gpu and n_threads is not None:
        # Set intra-op parallelism for CPU
        xla_flags = os.environ.get("XLA_FLAGS", "")

        # Add thread limit to XLA_FLAGS
        thread_flag = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n_threads}"

        if "intra_op_parallelism_threads" not in xla_flags:
            if xla_flags:
                xla_flags = f"{xla_flags} {thread_flag}"
            else:
                xla_flags = thread_flag

            os.environ["XLA_FLAGS"] = xla_flags
            logger.info(f"    JAX/XLA thread limit: {n_threads} threads")
            logger.debug(f"      XLA_FLAGS={xla_flags}")
        else:
            logger.debug("    XLA thread limit already configured")


def _configure_dask(n_threads: int = None):
    """
    Configure Dask thread limits.

    Parameters
    ----------
    n_threads : int, optional
        Number of threads for Dask
    """
    try:
        import dask
        import dask.config

        if n_threads is not None:
            # Configure Dask to use specified number of threads
            dask.config.set(scheduler="threads", num_workers=n_threads)
            logger.info(f"  Dask: thread limit set to {n_threads} threads")
            logger.debug(f"      Dask scheduler: threads, num_workers={n_threads}")
        else:
            logger.debug("  Dask: using default configuration")

    except ImportError:
        # Dask not installed, skip
        pass


def get_device_info():
    """
    Get information about available compute devices.

    Returns
    -------
    dict
        Dictionary with device information including:
        - gpu_available: bool
        - gpu_devices: list of device descriptions
        - cpu_count: int (logical cores)
        - jax_platform: str (current JAX platform)
    """
    info = {
        "gpu_available": False,
        "gpu_devices": [],
        "cpu_count": os.cpu_count(),
        "jax_platform": None,
    }

    try:
        import jax

        # Check current JAX platform
        try:
            current_backend = jax.devices()[0].platform
            info["jax_platform"] = current_backend
        except Exception:
            info["jax_platform"] = "unknown"

        # Check for GPU devices
        try:
            gpu_devices = jax.devices("gpu")
            if len(gpu_devices) > 0:
                info["gpu_available"] = True
                info["gpu_devices"] = [str(d) for d in gpu_devices]
        except RuntimeError:
            pass

    except ImportError:
        pass

    return info


def log_compute_environment():
    """Log information about the current compute environment."""
    info = get_device_info()

    logger.info("Compute Environment:")
    logger.info(f"  CPU cores: {info['cpu_count']}")
    logger.info(f"  JAX platform: {info['jax_platform']}")
    logger.info(f"  GPU available: {info['gpu_available']}")
    if info["gpu_available"]:
        logger.info(f"  GPU devices: {len(info['gpu_devices'])}")
        for i, device in enumerate(info["gpu_devices"]):
            logger.info(f"    {i}: {device}")
