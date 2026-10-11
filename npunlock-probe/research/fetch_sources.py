"""Fetch pinned, public Intel sources relevant to the NPU4000 port."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent / "sources"
compiler = json.loads((ROOT / "npu_compiler-tree.json").read_text())
driver = json.loads((ROOT / "linux-npu-driver-tree.json").read_text())
selected = []
for item in compiler["tree"]:
    p = item["path"]
    if item["type"] != "blob":
        continue
    if (p == ".gitmodules"
        or p.endswith(("vpu_nnrt_api.h", "vpu_nnrt_shavert.h"))
        or p.startswith("src/vpux_compiler/src/NPU40XX/dialect/NPUReg40XX/ops/act_")
        or p.startswith("src/vpux_compiler/src/conversion/rewriters/VPUASM2NPUReg40XX/act_")
        or p.startswith("src/vpux_compiler/src/conversion/rewriters/VPUMI40XX2VPUASM/kernel_")
        or p.startswith("src/vpux_compiler/src/dialect/VPUASM/ops/kernel_")
        or p in ("sw_runtime_kernels/kernels/cmake/mv_tools.cmake",
                 "sw_runtime_kernels/kernels/CMakeLists.txt",
                 "src/vpux_compiler/src/utils/sw_utils.cpp",
                 "sw_runtime_kernels/kernels/prebuild/act_shave_bin/activation_abs.3720xx.elf",
                 "sw_runtime_kernels/kernels/prebuild/act_shave_bin/activation_abs.4000xx.elf")):
        selected.append(("openvinotoolkit/npu_compiler", compiler["sha"], p))
for p in (".gitmodules", "firmware/include/api/vpu_nnrt_api.h", "firmware/include/api/vpu_nnrt_shavert.h"):
    selected.append(("intel/linux-npu-driver", driver["sha"], p))
for p in ("sw_runtime_kernels/kernels/inc/common_types.h",
          "src/vpux_compiler/include/vpux/compiler/NPU40XX/dialect/NPUReg40XX/firmware_headers/details/api/vpu_nnrt_common.h",
          "src/vpux_compiler/src/NPU40XX/dialect/NPUReg40XX/utils.cpp",
          "src/vpux_compiler/src/dialect/VPUASM/ops/declare_kernel_entry.cpp",
          "src/vpux_compiler/src/dialect/VPU/utils/sw_utils.cpp",
          "src/vpux_compiler/src/dialect/VPUIP/utils/sw_utils.cpp",
          "src/vpux_compiler/src/utils/llvm_to_binary.cpp"):
    selected.append(("openvinotoolkit/npu_compiler", compiler["sha"], p))
elf = json.loads((ROOT / "npu_compiler_elf-tree.json").read_text())
for p in ("core/include/vpux_elf/types/vpu_extensions.hpp", "loader/src/relocations.cpp",
          "loader/include/vpux_headers/relocations.hpp", "3rdparty/nnrt/details/api/vpu_nnrt_common.h",
          "3rdparty/nnrt/details/api/vpu_cmx_info_40xx.h", "loader/src/vpux_loader.cpp",
          "core/include/vpux_elf/reader.hpp"):
    selected.append(("openvinotoolkit/npu_compiler_elf", elf["sha"], p))

def fetch(record):
    repo, revision, path = record
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    response = requests.get(url, timeout=(15, 45))
    response.raise_for_status()
    data = response.content
    if data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        fields = dict(line.split(" ", 1) for line in data.decode().splitlines())
        url = f"https://media.githubusercontent.com/media/{repo}/{revision}/{path}"
        response = requests.get(url, timeout=(15, 45))
        response.raise_for_status()
        data = response.content
        assert len(data) == int(fields["size"]), "LFS artifact size mismatch"
        assert hashlib.sha256(data).hexdigest() == fields["oid"].removeprefix("sha256:"), "LFS artifact hash mismatch"
    target = ROOT / repo.split("/")[-1] / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"repo": repo, "revision": revision, "path": path, "url": url,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

with ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(fetch, selected))
(ROOT / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(f"Fetched {len(records)} pinned source/artifact files.")
