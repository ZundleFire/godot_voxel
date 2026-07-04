"""
ISPC builder for the voxel module's SCons build.

Provides an SCons Tool that compiles .ispc files to .obj (Windows) or .o (Linux/macOS).
ISPC (Intel SPMD Program Compiler) generates SIMD-optimized code from a C-like SPMD language.

Usage in SCsub:
    import ispc_builder
    ispc_builder.setup_ispc(env_voxel, env)
    # Then .ispc files listed in sources will be compiled automatically.

The builder auto-detects ISPC in PATH, or uses the ISPC_PATH environment variable.
If ISPC is not found, the build proceeds without ISPC support (scalar fallback).
"""

import os
import subprocess
from SCons.Builder import Builder
from SCons.Action import Action


def _find_ispc():
    """Try to find the ISPC compiler."""
    # Check environment variable first
    ispc_path = os.environ.get("ISPC_PATH", "")
    if ispc_path:
        exe = os.path.join(ispc_path, "ispc.exe") if os.name == "nt" else os.path.join(ispc_path, "ispc")
        if os.path.isfile(exe):
            return exe

    # Check PATH
    try:
        result = subprocess.run(
            ["ispc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "ispc"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check common install locations on Windows
    if os.name == "nt":
        common_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\ispc"),
            os.path.expandvars(r"%ProgramFiles%\ISPC"),
            os.path.expandvars(r"%ProgramFiles(x86)%\ISPC"),
            r"C:\ispc",
            r"C:\DEV_DRIVE\ispc",
            r"F:\ispc",
        ]
        for p in common_paths:
            exe = os.path.join(p, "bin", "ispc.exe")
            if os.path.isfile(exe):
                return exe
            exe = os.path.join(p, "ispc.exe")
            if os.path.isfile(exe):
                return exe

    return None


def _ispc_emitter(target, source, env):
    """
    Emitter for ISPC: besides the dispatch .obj, ISPC generates:
    - A _ispc.h header for extern "C" declarations
    - Per-target .obj files (e.g. _sse4.obj, _avx2.obj) that contain the
      actual SIMD implementations. These MUST be linked alongside the dispatch obj.
    """
    base = os.path.splitext(str(source[0]))[0]
    obj_suffix = ".obj" if os.name == "nt" else ".o"
    header = base + "_ispc.h"

    # Parse targets to find per-target object files
    ispc_targets = env.get("ISPC_TARGETS", "sse4-i32x4,avx2-i32x8")
    target_list = [t.strip() for t in ispc_targets.split(",")]

    extra_objs = []
    for t in target_list:
        # ISPC names per-target files with a suffix derived from the target,
        # e.g. "sse4-i32x4" -> "_sse4", "avx2-i32x8" -> "_avx2"
        suffix = t.split("-")[0]  # "sse4", "avx2", etc.
        per_target_obj = base + "_" + suffix + obj_suffix
        extra_objs.append(per_target_obj)
        per_target_header = base + "_ispc_" + suffix + ".h"
        env.Clean(target, per_target_header)

    # Add per-target objects as additional targets so SCons links them
    for obj_path in extra_objs:
        target.append(obj_path)

    env.SideEffect(header, target[0])
    env.Clean(target, header)
    return target, source


def _ispc_action_string(source, target, env, for_signature=False):
    """Generate the ISPC command string for display."""
    return "Compiling ISPC: %s" % os.path.basename(str(source[0]))


def _ispc_action(target, source, env):
    """Build action that invokes ISPC."""
    ispc_exe = env.get("ISPC_COMPILER", "ispc")
    src = str(source[0])
    obj = str(target[0])

    # Generate header path (same dir as source, with _ispc.h suffix)
    base = os.path.splitext(src)[0]
    header = base + "_ispc.h"

    cmd = [
        ispc_exe,
        src,
        "-o", obj,
        "-h", header,
    ]

    # Position-independent code (not needed on Windows)
    if os.name != "nt":
        cmd += ["--pic"]

    # Add target ISA flags
    ispc_targets = env.get("ISPC_TARGETS", "sse4-i32x4,avx2-i32x8")
    cmd += ["--target=" + ispc_targets]

    # Architecture
    arch = env.get("ISPC_ARCH", "x86-64")
    cmd += ["--arch=" + arch]

    # Optimization level
    if env.get("ISPC_OPT", "") == "fast":
        cmd += ["-O2", "--opt=fast-math"]
    else:
        cmd += ["-O2"]

    # Include paths
    ispc_includes = env.get("ISPC_INCLUDES", [])
    for inc in ispc_includes:
        cmd += ["-I", inc]

    # Warning flags
    cmd += ["--wno-perf"]

    # Windows-specific: use MSVC-compatible obj format
    if os.name == "nt":
        cmd += ["--target-os=windows"]

    print("  ISPC: %s -> %s" % (os.path.basename(src), os.path.basename(obj)))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ISPC Error:")
        print(result.stderr)
        return result.returncode

    if result.stderr:
        # Print warnings but don't fail
        for line in result.stderr.splitlines():
            if "warning" in line.lower():
                print("  ISPC Warning: " + line)

    return 0


def setup_ispc(env_voxel, env):
    """
    Set up ISPC compilation in the voxel module build environment.

    Returns True if ISPC is available and configured, False otherwise.
    The caller should check this and conditionally add .ispc source files
    and set VOXEL_ISPC_ENABLED define.
    """
    ispc_exe = _find_ispc()
    if ispc_exe is None:
        print("voxel: ISPC compiler not found — SIMD noise will use scalar fallback")
        return False

    # Verify it works
    try:
        result = subprocess.run(
            [ispc_exe, "--version"],
            capture_output=True, text=True, timeout=5
        )
        version_str = result.stdout.strip() if result.stdout else "unknown"
        print("voxel: Found ISPC — %s" % version_str)
    except Exception as e:
        print("voxel: ISPC found but failed to run: %s" % e)
        return False

    # Store compiler path
    env_voxel["ISPC_COMPILER"] = ispc_exe

    # Set default targets based on platform
    # SSE4 as baseline (covers most x86_64), AVX2 for modern CPUs
    if not env_voxel.get("ISPC_TARGETS"):
        env_voxel["ISPC_TARGETS"] = "sse4-i32x4,avx2-i32x8"

    if not env_voxel.get("ISPC_ARCH"):
        env_voxel["ISPC_ARCH"] = "x86-64"

    # Set include paths for ISPC headers
    # The .isph files live alongside the .ispc files
    voxel_dir = os.path.dirname(os.path.abspath(__file__))
    env_voxel["ISPC_INCLUDES"] = [
        os.path.join(voxel_dir, "util", "noise"),
    ]

    # Object file suffix
    obj_suffix = ".obj" if os.name == "nt" else ".o"

    # Create the ISPC builder
    ispc_builder = Builder(
        action=Action(_ispc_action, _ispc_action_string),
        suffix=obj_suffix,
        src_suffix=".ispc",
        emitter=_ispc_emitter,
        single_source=True,
    )

    env_voxel.Append(BUILDERS={"ISPC": ispc_builder})

    # Define preprocessor macro so C++ code knows ISPC is available
    env_voxel.Append(CPPDEFINES=["VOXEL_ISPC_ENABLED"])

    return True
