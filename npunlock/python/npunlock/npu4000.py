"""Experimental Lunar Lake FP16/FP32 patching, without changing NPU3720 APIs.

 Supports verified NPU4000 Abs/Power/Convert/Select carriers and small 4000xx kernels with no
kernel data. Keeps the compiler's DMA, padding, scheduling, and code windows.
"""
from dataclasses import replace
import hashlib
import json
from math import prod
import struct
import xml.etree.ElementTree as ET

NO_FILE_DATA = {8, 0x8AAAAAAC, 0x8AAAAAAD}
ABS_TEXT_SHA256 = "5b51680501c037dfeb9fe4653df9d453fd5b42224681b41338b6de591d225a9d"
POWER_TEXT_SHA256 = "0b433416990bbef1a42052e60cbd3bacd9b0c5f87677a3b85cb8df7579fb50b6"
CAST_TEXT_SHA256 = "70fbdfd15d4a0a2785a73576279d6a993eea55387150ea0b80e2d40a95c33ddc"
SELECT_TEXT_SHA256 = "2bb6d89b2a4d94c392058f5b15e77418d68891e6e060e56feaf19a6040774af7"
REDUCE_TEXT_SHA256 = {
    "ReduceSum": "043a9d9dd481d9bf63ed78c7344a6a5faee8eab024e052ff62d885c87bb2a561",
    "ReduceMean": "b4d29126e77d745fa43305cb23f6dc3e5877ed44b8648dc03495e576be413dba",
}
MAX_ELEMENTS = 65536
MAX_TENSOR_ELEMENTS = 1048576  # Existing frozen DPU adapters retain their validated bounds.
MAX_STORAGE_ELEMENTS = 16777216  # Legacy operator workspaces, including float metadata packets.
MAX_ALLOCATION_BYTES = 1024 * 1024 * 1024
MAX_DMA_ELEMENTS = MAX_ALLOCATION_BYTES // 4
MAX_STORAGE_RANK = 64


def compile_byte_packet(header, sizes, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """Pack static U32 metadata and raw device bytes into a complete SHAVE frame."""
    import numpy as np
    from . import Graph, input, op, constant
    header, sizes = tuple(header), tuple(sizes)
    if (not header or any(type(v) is not int or not 0 <= v < 2**32 for v in header) or
            not sizes or any(type(v) is not int or v < 1 for v in sizes)):
        raise ValueError('Byte packet requires U32 metadata and positive payload sizes')
    total = 4*len(header)+sum(sizes)
    words = (max(total, header[15] if len(header)==16 else 0)+7)//8*2
    if words > MAX_ELEMENTS:
        raise ValueError('Byte packet exceeds the complete SHAVE carrier')
    inputs = [input(f'x{i}', shape=(1, size), dtype='u8') for i, size in enumerate(sizes)]
    parts = [constant(np.array(header, dtype='<u4').view(np.uint8).reshape(1, -1)), *inputs]
    if words*4 > total:
        parts.append(constant(np.zeros((1, words*4-total), dtype=np.uint8)))
    output = op('Concat', *parts, axis=1, _shape=(1, words*4), _dtype='u8', _name='y')
    return compile_precision_graph(Graph(inputs, [output]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)
RELOCATION_NAMES = {0: "R_VPU_64", 4: "R_VPU_32", 17: "R_VPU_32_BIT_OR_B21_B26_UNSET",
                    18: "R_VPU_64_BIT_OR_B21_B26_UNSET"}


def bounded(data, offset, size):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ValueError("File-backed extent out of bounds")
    return data[offset:offset + size]


def string_at(data, offset):
    if offset < 0 or offset >= len(data):
        raise ValueError("String offset out of bounds")
    end = data.find(b"\0", offset)
    if end < 0:
        raise ValueError("String has no terminator")
    return data[offset:end].decode(errors="replace")


class Elf:
    def __init__(self, data):
        self.data = data
        header = bounded(data, 0, 64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("Requires ELF64 little endian")
        shoff = struct.unpack_from("<Q", header, 40)[0]
        entry_size, count, names_index = struct.unpack_from("<HHH", header, 58)
        if entry_size != 64 or not count or names_index >= count:
            raise ValueError("Unsupported section table")
        table = bounded(data, shoff, count * 64)
        self.sections = []
        for i in range(count):
            n, t, f, a, o, s, link, info, align, entsize = struct.unpack_from("<IIQQQQIIQQ", table, i * 64)
            if t not in NO_FILE_DATA:
                bounded(data, o, s)
            self.sections.append(dict(index=i, name_offset=n, type=t, flags=f, address=a,
                                      offset=o, size=s, link=link, info=info, align=align, entsize=entsize))
        names = self.content(self.sections[names_index])
        for section in self.sections:
            section["name"] = string_at(names, section["name_offset"])
        self.named = {s["name"]: s for s in self.sections}
        if len(self.named) != len(self.sections):
            raise ValueError("Duplicate ELF section names are unsupported")
        self.relocations = []
        for section in self.sections:
            if section["type"] != 4:
                continue
            if section["entsize"] != 24 or section["size"] % 24:
                raise ValueError("Unsupported RELA table")
            if section["link"] >= count or section["info"] >= count:
                raise ValueError("RELA link or target section out of bounds")
            symbols = self.sections[section["link"]]
            if (symbols["type"] != 2 or symbols["entsize"] != 24 or symbols["size"] % 24
                or symbols["link"] >= count or self.sections[symbols["link"]]["type"] != 3):
                raise ValueError("Unsupported symbol table")
            symbol_data = self.content(symbols)
            symbol_strings = self.content(self.sections[symbols["link"]])
            for offset in range(0, section["size"], 24):
                target_offset, info, addend = struct.unpack_from("<QQq", self.content(section), offset)
                symbol_index, kind = info >> 32, info & 0xFFFFFFFF
                name, _, _, shndx, value, size = struct.unpack("<IBBHQQ", bounded(symbol_data, symbol_index * 24, 24))
                source_section = self.sections[shndx]["name"] if shndx < len(self.sections) else f"special:{shndx:#x}"
                self.relocations.append(dict(target_section=self.sections[section["info"]]["name"],
                    target_offset=target_offset, type=kind, type_name=RELOCATION_NAMES.get(kind, f"unknown:{kind}"),
                    symbol_name=string_at(symbol_strings, name), symbol_section=source_section,
                    symbol_value=value, symbol_size=size, addend=addend))

    def content(self, section):
        if section["type"] in NO_FILE_DATA:
            raise ValueError("Section has no file data")
        return bounded(self.data, section["offset"], section["size"])

    def relocation(self, section, offset):
        matches = [r for r in self.relocations if r["target_section"] == section and r["target_offset"] == offset]
        if len(matches) != 1:
            raise ValueError(f"Expected one relocation: {section}+{offset:#x}")
        return matches[0]

    def memref(self, offset):
        params = self.content(self.named["shave.params"])
        raw = bounded(params, offset, 40)
        addr, static, rank, _, _, dtype, order, location = struct.unpack("<QIIIIIQI", raw)
        if not static or not 1 <= rank <= 15:
            raise ValueError("Only static MemRefData supported by this inspection")
        dims_reloc = self.relocation("shave.params", offset + 16)
        strides_reloc = self.relocation("shave.params", offset + 20)
        for reloc in (dims_reloc, strides_reloc):
            if reloc["symbol_section"] != "shave.params" or reloc["type"] != 4:
                raise ValueError("Unexpected dims/strides relocation")
        dims = struct.unpack("<" + "i" * rank, bounded(params, dims_reloc["symbol_value"] + dims_reloc["addend"], 4 * rank))
        strides = struct.unpack("<" + "q" * rank, bounded(params, strides_reloc["symbol_value"] + strides_reloc["addend"], 8 * rank))
        if any(d <= 0 for d in dims):
            raise ValueError("Non-positive static dimension")
        data_relocs = [r for r in self.relocations if r["target_section"] == "shave.params" and r["target_offset"] == offset]
        if not data_relocs and location != 2:
            raise ValueError("Missing data relocation outside CMX")
        return dict(offset=offset, unrelocated_data=hex(addr), static=static, rank=rank,
                    dtype=dtype, order=hex(order), location=location, dims=dims, strides_bits=strides,
                    elements=prod(dims), data_relocations=data_relocs)


def inspect(data, *, memref_count=2, kernel_sha256=None):
    if memref_count not in (2, 3, 4):
        raise ValueError("Inspection supports two, three or four tensor records")
    elf = Elf(data)
    ranges, invocations = [], []
    sections=elf.sections if kernel_sha256 is None else sorted(elf.sections,key=lambda s:not s['name'].startswith('task.shave.range.'))
    for section in sections:
        name = section["name"]
        if name.startswith("task.shave.range."):
            if section["size"] % 40:
                raise ValueError("Range section is not a multiple of 40 bytes")
            for base in range(0, section["size"], 40):
                raw = bounded(elf.content(section), base, 40)
                record = dict(section=name, index=base // 40, workload_type=raw[0],
                              kernel_entry=hex(struct.unpack_from("<Q", raw, 8)[0]),
                              code_size=struct.unpack_from("<I", raw, 24)[0],
                              invocation_count=struct.unpack_from("<I", raw, 32)[0])
                if raw[0] == 0:
                    record["text_relocation"] = elf.relocation(name, base + 16)
                if kernel_sha256 is not None:
                    reloc=record.get('text_relocation',{})
                    if reloc.get('symbol_section')!='shave.text' or reloc.get('type')!=0:continue
                    code=bounded(elf.content(elf.named['shave.text']),reloc['symbol_value']+reloc['addend'],record['code_size'])
                    if hashlib.sha256(code).hexdigest()!=kernel_sha256:continue
                ranges.append(record)
        elif name.startswith("task.shave.invocation."):
            if section["size"] % 96:
                raise ValueError("Invocation section is not a multiple of 96 bytes")
            for base in range(0, section["size"], 96):
                raw = bounded(elf.content(section), base, 96)
                range_relocs = [r for r in elf.relocations if r["target_section"] == name and r["target_offset"] == base]
                record = dict(section=name, index=base // 96,
                              tile=struct.unpack_from("<I", raw, 84)[0],
                              range_index=struct.unpack_from("<I", raw, 88)[0],
                              unrelocated_range_pointer=hex(struct.unpack_from("<Q", raw, 0)[0]),
                              range_relocations=range_relocs)
                if kernel_sha256 is not None:
                    tile,engine=map(int,name.removeprefix('task.shave.invocation.').split('.'))
                    if not any(r['section']==f'task.shave.range.{tile}.{engine}' and r['index']==record['range_index'] for r in ranges):continue
                params = [r for r in elf.relocations if r["target_section"] == name and r["target_offset"] == base + 8]
                if params:
                    params_reloc = elf.relocation(name, base + 8)
                    if params_reloc["symbol_section"] != "shave.params" or params_reloc["type"] != 0:
                        raise ValueError("Unexpected kernel_args relocation")
                    param_base = params_reloc["symbol_value"] + params_reloc["addend"]
                    record["params_relocation"] = params_reloc
                    record["memrefs"] = [elf.memref(param_base + i * 40) for i in range(memref_count)]
                    if memref_count == 2:
                        record["unary_memrefs"] = record["memrefs"]
                invocations.append(record)
    return dict(blob_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                no_file_data_sections=[s for s in elf.sections if s["type"] in NO_FILE_DATA],
                ranges=ranges, invocations=invocations, relocations=elf.relocations)


def kernel_text(elf32):
    """Validate a linked SHAVE ELF32; the caller must compile for 4000xx."""
    header = bounded(elf32, 0, 52)
    if header[:6] != b"\x7fELF\x01\x01" or struct.unpack_from("<HH", header, 16) != (2, 2):
        raise ValueError("Requires executable little-endian SHAVE/SPARC ELF32")
    entry, shoff = struct.unpack_from("<I", header, 24)[0], struct.unpack_from("<I", header, 32)[0]
    size, count, names_index = struct.unpack_from("<HHH", header, 46)
    if size != 40 or not count or names_index >= count:
        raise ValueError("Unsupported ELF32 section table")
    table = bounded(elf32, shoff, count * size)
    sections = [struct.unpack_from("<10I", table, i * size) for i in range(count)]
    names_header = sections[names_index]
    names = bounded(elf32, names_header[4], names_header[5])
    named = {string_at(names, h[0]): h for h in sections}
    if len(named) != count or ".text" not in named or ".arg.data" not in named:
        raise ValueError("Missing or duplicate kernel sections")
    text, arg = named[".text"], named[".arg.data"]
    if text[1] != 1 or text[3] != 0x1D000000 or entry != text[3]:
        raise ValueError("Unexpected kernel entry/text address")
    if arg[3] != 0x1E000000 or arg[5] != 0:
        raise ValueError("Kernel data is not supported by in-place patching")
    for h in sections:
        if h[1] != 8:
            bounded(elf32, h[4], h[5])
        if (h[1] in (4, 9) and h[5]) or (h[2] & 2 and h[5] and h is not text):
            raise ValueError("Unresolved relocations or extra allocated kernel sections")
    return bounded(elf32, text[4], text[5])


def _dense_float(ref, tile, elf, dtype):
    dtype_id, itemsize = {"f16": (2, 2), "f32": (1, 4)}[dtype]
    if ref["static"] != 1 or ref["dtype"] != dtype_id or ref["location"] != 2 or ref["data_relocations"]:
        raise ValueError(f"Requires static {dtype} MemRefs with fixed CMX addresses; precision lowering rejected")
    if any(s < 0 for s in ref["strides_bits"]) or ref["elements"] > 0xFFFFFFFF:
        raise ValueError("Unsupported tensor extent/stride")
    expected = itemsize * 8
    for stride, dimension in sorted(zip(ref["strides_bits"], ref["dims"])):
        if dimension > 1:
            if stride != expected:
                raise ValueError("Non-dense MemRef layout")
            expected *= dimension
    address = int(ref["unrelocated_data"], 16)
    if not 0 <= tile <= 3:
        raise ValueError("Unsupported CMX tile index")
    # NPU4000 encodes the CMX tile selection as a one-hot bit at 21+tile.
    region = 0x40000000 + (1 << (21 + tile))
    workspace = elf.named.get(f"buffer.CMX_NN.{tile}")
    span = ref["elements"] * itemsize
    if (workspace is None or workspace["type"] != 0x8AAAAAAD
        or address % itemsize or address < region or address + span > region + workspace["size"]):
        raise ValueError("MemRef is outside the observed tile CMX workspace")
    return address, span


def patch_abs_carrier(graph_blob, kernel_elf, *, vendor_id, device_id, compiler_version, dtype="f16"):
    """Replace the shared Abs code window after validating the Lunar Lake profile.

    This affects all Abs tasks using that window. Use compile_unary() for the
    supported single-op graph. No ELF growth, multi-node selection, or DDR ABI.
    """
    return _patch_carrier(graph_blob, kernel_elf, vendor_id=vendor_id, device_id=device_id,
                          compiler_version=compiler_version, dtype=dtype, carrier="Abs", input_count=1)


def patch_binary_carrier(graph_blob, kernel_elf, *, vendor_id, device_id, compiler_version, dtype):
    """Patch a verified Power carrier with two matching dense floating inputs."""
    return _patch_carrier(graph_blob, kernel_elf, vendor_id=vendor_id, device_id=device_id,
                          compiler_version=compiler_version, dtype=dtype, carrier="Power", input_count=2)


def patch_kv_select(graph_blob, kernel_elf, *, vendor_id, device_id, compiler_version):
    """Replace only the verified FP16 KV-row/mask Select code on Lunar Lake.

    All invocations sharing the code window must satisfy the bounded layout;
    unknown profiles, absent carriers and incompatible layouts are rejected.
    """
    return _patch_carrier(graph_blob, kernel_elf, vendor_id=vendor_id, device_id=device_id,
        compiler_version=compiler_version, dtype='f16', carrier='Select', input_count=3, select_kernel=True)


def _kv_select_layout(refs):
    if any(r['rank']!=4 or r['order']!='0x4321' or r['elements']>MAX_ELEMENTS or
           any(s%16 or not 0<=s//16<=MAX_ELEMENTS for s in r['strides_bits']) for r in refs):
        raise ValueError('KV Select requires bounded rank-four FP16 layouts')
    condition,current,past,out=refs; width,rows,heads,n=out['dims']
    if width!=64 or not 1<=rows<=64 or not 1<=heads<=24 or n!=1:
        raise ValueError('KV Select requires 64-element complete rows, <=64 rows and <=24 heads')
    mask=(rows==heads==1 and condition['dims']==out['dims'] and
          current['dims']==past['dims']==(1,1,1,1))
    if mask:
        if condition['strides_bits'][0]!=16 or out['strides_bits'][0]!=16:
            raise ValueError('Mask Select requires contiguous FP16 conditions and output')
        return
    if (condition['dims']!=(1,rows,1,1) or current['dims']!=(64,1,heads,1) or past['dims']!=out['dims'] or
        condition['strides_bits'][1]!=16 or current['strides_bits'][0]!=16 or current['strides_bits'][2]!=1024 or
        any(r['strides_bits'][:3]!=(16,1024,1024*rows) for r in (past,out))):
        raise ValueError('KV Select requires the verified condition/current/past broadcast and row strides')


def _patch_carrier(graph_blob, kernel_elf, *, vendor_id, device_id, compiler_version, dtype, carrier, input_count,
                   output_dtype=None, reduction_count=None, select_kernel=False):
    if dtype not in ("f16", "f32"):
        raise ValueError("Only f16/f32 custom kernels are supported")
    output_dtype = output_dtype or dtype
    if output_dtype not in ("f16", "f32"):
        raise ValueError("Only f16/f32 outputs are supported")
    if (vendor_id, device_id, tuple(compiler_version)) != (0x8086, 0x643E, (8, 3)):
        raise ValueError("Unsupported device/compiler profile for NPU4000 unary patching")
    elf = Elf(graph_blob)
    fingerprint = {"Abs": ABS_TEXT_SHA256, "Power": POWER_TEXT_SHA256, "Convert": CAST_TEXT_SHA256, "Select": SELECT_TEXT_SHA256,
                   **REDUCE_TEXT_SHA256}[carrier]
    evidence = inspect(graph_blob, memref_count=input_count + 1,kernel_sha256=fingerprint if select_kernel else None)
    code = kernel_text(kernel_elf)
    text = elf.named.get("shave.text")
    if text is None or text["type"] != 1 or not evidence["ranges"] or not evidence["invocations"]:
        raise ValueError("Missing NPU4000 unary carrier sections/tasks")
    if select_kernel:
        for task in evidence['ranges']:
            count=sum(t['section'].replace('invocation','range')==task['section'] and t['range_index']==task['index'] for t in evidence['invocations'])
            # Link-time counters are zero in this compiler profile. Require a
            # static invocation, and check the counter when explicitly populated.
            if not count or task['invocation_count'] not in (0,count):
                raise ValueError('Selected carrier invocation count differs from its range')
    windows = set()
    range_counts = {}
    for task in evidence["ranges"]:
        reloc = task.get("text_relocation", {})
        if (task["workload_type"] != 0 or task["kernel_entry"] != "0x1d000000"
            or reloc.get("type") != 0 or reloc.get("symbol_section") != "shave.text"):
            raise ValueError("Unsupported NPU4000 range/text contract")
        windows.add((reloc["symbol_value"] + reloc["addend"], task["code_size"]))
        range_counts[task["section"]] = elf.named[task['section']]['size']//40
    if len(windows) != 1:
        raise ValueError("Expected one shared unary code window")
    offset, extent = windows.pop()
    original_code = bounded(elf.content(text), offset, extent)
    if hashlib.sha256(original_code).hexdigest() != fingerprint:
        raise ValueError(f"Carrier code does not match the verified NPU4000 {carrier} kernel")
    if not 0 < len(code) <= extent:
        raise ValueError("Replacement kernel exceeds the existing code window")
    pointer_bases = {}
    contracts = []
    for task in evidence["invocations"]:
        try:
            tile, engine = map(int, task["section"].removeprefix("task.shave.invocation.").split("."))
        except ValueError as error:
            raise ValueError("Unsupported invocation section identity") from error
        range_section = f"task.shave.range.{tile}.{engine}"
        if task["tile"] != tile or not 0 <= task["range_index"] < range_counts.get(range_section, 0):
            raise ValueError("Invocation references an invalid tile/range")
        if task["range_relocations"]:
            if len(task["range_relocations"]) != 1:
                raise ValueError("Ambiguous invocation range reference")
            reloc = task["range_relocations"][0]
            if (reloc["type"] != 0 or reloc["symbol_section"] != range_section
                or reloc["symbol_value"] + reloc["addend"] != task["range_index"] * 40):
                raise ValueError("Unexpected invocation range relocation")
        else:
            pointer = int(task["unrelocated_range_pointer"], 16)
            base = pointer - task["range_index"] * 40
            if not 0 < base < 0x100000000 or pointer_bases.setdefault(range_section, base) != base:
                raise ValueError("Inconsistent fixed range pointer")
        refs = task.get("memrefs", [])
        if len(refs) != input_count + 1:
            raise ValueError("Unexpected carrier MemRef count")
        left, right = refs[0], refs[-1]
        if reduction_count is not None:
            if (input_count != 1 or dtype != "f32" or output_dtype != "f32" or
                left["rank"] != right["rank"] or left["rank"] > 4 or left["order"] != right["order"] or
                any(b not in (1, a) for a, b in zip(left["dims"], right["dims"])) or
                left["elements"] != right["elements"] * reduction_count or
                left["elements"] > MAX_ELEMENTS):
                raise ValueError("Reduction carrier has an unsupported layout or partial reduction")
        elif carrier=='Select':
            _kv_select_layout(refs)
        elif any(left[k] != ref[k] for ref in refs[1:] for k in ("dims", "order", "elements")):
            raise ValueError("Input/output unary layouts do not match")
        dtypes = [dtype] * input_count + [output_dtype]
        strides = [tuple(s // (16 if dt == "f16" else 32) for s in ref["strides_bits"])
                   for ref, dt in zip(refs, dtypes)]
        if carrier!='Select' and reduction_count is None and any(s != strides[0] for s in strides[1:]):
            raise ValueError("Input/output layouts do not match")
        spans = [_dense_float(ref, tile, elf, dt) for ref, dt in zip(refs, dtypes)]
        if carrier=='Select' and any(address%16 for address,_ in spans):
            raise ValueError('KV Select requires 16-byte aligned CMX buffers')
        output, output_span = spans[-1]
        if any(max(source, output) < min(source + span, output + output_span) for source, span in spans[:-1]):
            raise ValueError("Overlapping input/output CMX buffers")
        contracts.append({"tile": tile, "elements": right["elements"] if carrier=='Select' else left["elements"],
                          "span_bytes": output_span, "dtype": dtype})
    start = text["offset"] + offset
    patched = bytearray(graph_blob)
    patched[start:start + len(code)] = code
    if len(patched) != len(graph_blob) or patched[:start] != graph_blob[:start] or patched[start + len(code):] != graph_blob[start + len(code):]:
        raise ValueError("Unexpected modification outside the kernel code")
    report = {"schema": "npunlock.npu4000.unary-inplace.v1", "device_id": hex(device_id),
              "compiler_version": list(compiler_version), "original_sha256": evidence["sha256"],
              "patched_sha256": hashlib.sha256(patched).hexdigest(),
              "kernel_elf_sha256": hashlib.sha256(kernel_elf).hexdigest(),
              "code_offset": start, "code_bytes": len(code), "window_bytes": extent,
              "carrier": carrier, "dtype": dtype, "output_dtype": output_dtype, "input_count": input_count,
              "changed_bytes": sum(a != b for a, b in zip(graph_blob, patched)),
              "ranges": len(evidence["ranges"]), "invocations": len(contracts), "contracts": contracts}
    return bytes(patched), report


def compile_unary(shape, kernel_elf, *, dtype="f16", native_dir=None, timeout_ms=20_000, libraries=None):
    """Compile a static FP16/FP32 Abs carrier and patch its shared code window.

    kernel_elf must be linked for 4000xx. The returned Program uses copied
    inference normally. No OpenVINO Python/runtime package is required.
    """
    from . import Graph, input, op
    if not isinstance(kernel_elf, bytes):
        raise TypeError("kernel_elf must be linked ELF bytes")
    if dtype not in ("f16", "f32"):
        raise ValueError("Only f16/f32 custom kernels are supported")
    x = input("x", shape=shape, dtype=dtype)
    if not 1 <= len(x.shape) <= 4 or prod(x.shape) > MAX_ELEMENTS:
        raise ValueError("Experimental unary carrier requires 1-4 dimensions and at most 65536 elements")
    y = op("Abs", x, _shape=x.shape, _dtype=x.dtype, _name="y")
    program = compile_precision_graph(Graph([x], [y], name="npu4000_unary"), native_dir=native_dir,
                                      timeout_ms=timeout_ms, libraries=libraries)
    info = program.ir_provenance
    blob, report = patch_abs_carrier(program.graph_blob, kernel_elf, vendor_id=info.vendor_id,
                                    device_id=info.device_id, compiler_version=info.compiler_version, dtype=dtype)
    report["shape"] = list(x.shape)
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def compile_binary(shape, kernel_elf, *, dtype="f32", native_dir=None, timeout_ms=20_000, libraries=None):
    """Compile a two-input Power carrier for a same-shape custom binary kernel."""
    from . import Graph, input, op
    if dtype not in ("f16", "f32") or not isinstance(kernel_elf, bytes):
        raise ValueError("Requires f16/f32 and linked kernel bytes")
    x = input("x", shape=shape, dtype=dtype)
    if not 1 <= len(x.shape) <= 4 or prod(x.shape) > MAX_ELEMENTS:
        raise ValueError("Binary carrier requires 1-4 dimensions and at most 65536 elements")
    b = input("b", shape=shape, dtype=dtype)
    y = op("Power", x, b, _shape=x.shape, _dtype=dtype, _name="y")
    program = compile_precision_graph(Graph([x, b], [y], name="npu4000_binary"), native_dir=native_dir,
                                      timeout_ms=timeout_ms, libraries=libraries)
    info = program.ir_provenance
    blob, report = patch_binary_carrier(program.graph_blob, kernel_elf, vendor_id=info.vendor_id,
        device_id=info.device_id, compiler_version=info.compiler_version, dtype=dtype)
    report["shape"] = list(x.shape)
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def compile_int64_counter(kernel_elf, *, native_dir=None, timeout_ms=20_000, libraries=None):
    """One complete two-word INT64 add; F32 is a raw-bit transport carrier."""
    from . import Graph, input, op
    if not isinstance(kernel_elf, bytes):
        raise TypeError('INT64 counter kernel must be linked ELF bytes')
    x, b = (input(name, shape=(2,), dtype='f32') for name in ('x', 'b'))
    y = op('Power', x, b, _shape=(2,), _dtype='f32', _name='y')
    program = compile_precision_graph(Graph([x, b], [y], name='int64_counter'),
        native_dir=native_dir, timeout_ms=timeout_ms, libraries=libraries, reference_sw=True)
    evidence = inspect(program.graph_blob, memref_count=3)
    if (len(evidence['invocations']) != 1 or len(evidence['ranges']) != 1 or
            any(ref['elements'] != 2 for task in evidence['invocations'] for ref in task.get('memrefs', []))):
        raise ValueError('INT64 counter carrier must keep both words in a single invocation')
    info = program.ir_provenance
    blob, report = patch_binary_carrier(program.graph_blob, kernel_elf, vendor_id=info.vendor_id,
        device_id=info.device_id, compiler_version=info.compiler_version, dtype='f32')
    report.update(operation='int64_add_raw_words', scheduling='single_complete_pair', shape=[2])
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def compile_classification_rows(shape, kernel_elf, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """FP32 Power carrier keeping each classification row intact in one task."""
    from . import Graph, input, op
    shape = tuple(shape)
    if (len(shape) != 2 or any(type(d) is not int or d < 1 for d in shape) or
            prod(shape) > MAX_ELEMENTS or not isinstance(kernel_elf, bytes)):
        raise ValueError('Classification carrier requires bounded complete rows and linked ELF bytes')
    x, b = (input(name, shape=shape, dtype='f32') for name in ('x', 'b'))
    y = op('Power', x, b, _shape=shape, _dtype='f32', _name='y')
    program = compile_precision_graph(Graph([x, b], [y], name='classification_rows'), libraries=libraries,
        native_dir=native_dir, timeout_ms=timeout_ms, reference_sw=True)
    evidence = inspect(program.graph_blob, memref_count=3)
    rows, width = shape
    layouts = {(width, rows, 1, 1)}
    if rows == 1:
        layouts.add((1, 1, width, 1))
    if (len(evidence['ranges']) != 1 or len(evidence['invocations']) != 1 or
        any(len(task.get('memrefs', [])) != 3 or
            any(tuple(ref['dims']) not in layouts or ref['elements'] != prod(shape) for ref in task['memrefs'])
            for task in evidence['invocations'])):
        actual = [[ref.get('dims') for ref in task.get('memrefs', [])] for task in evidence['invocations']]
        raise ValueError(f'Classification carrier splits a row or changes its verified layout: {shape}, {actual}')
    info = program.ir_provenance
    blob, report = patch_binary_carrier(program.graph_blob, kernel_elf, vendor_id=info.vendor_id,
        device_id=info.device_id, compiler_version=info.compiler_version, dtype='f32')
    report.update(operation='classification_rows', shape=list(shape), scheduling='single_complete_rows')
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def mask_packet_spec(count, kind, code):
    """Raw byte packet ABI: four F32 metadata words, payloads, 8-byte padding."""
    if (any(type(v) is not int for v in (count, kind, code)) or not 1 <= count <= MAX_ELEMENTS or
            not 0 <= kind <= 12 or code not in (0, 1, 2, 3) or kind >= 9 and code != 3):
        raise ValueError('Invalid mask packet count/operator/storage code')
    width = (4, 2, 8, 1)[code]
    sizes = ((count, count*width, count*width) if kind == 6 else (count,) if kind in (8, 9) else
             (count*width,) if kind == 7 else (count*width, count*width))
    output = count*width if kind in (6, 8) else count
    size = (max(16+sum(sizes), output)+7)//8*8
    if size > MAX_ELEMENTS*4:
        raise ValueError('Mask packet exceeds the complete SHAVE carrier bound')
    return sizes, output, size//4


def rng_packet_words(count):
    if type(count) is not int or not 1 <= count <= MAX_ELEMENTS:
        raise ValueError('RNG packet requires 1..65536 elements')
    return (max(count, 16)+1)//2*2


def compile_rng_packet(count, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """Dynamic seed/counter header, padded by byte DMA; no random host payload."""
    import numpy as np
    from . import Graph, input, op, constant
    words = rng_packet_words(count)
    header = input('header', shape=(1, 32), dtype='u8')
    # Always enlarge the input: an identity graph has ambiguous input/output bindings.
    y = op('Concat', header, constant(np.zeros((1, words*4-32), dtype=np.uint8)),
           axis=1, _shape=(1, words*4), _dtype='u8', _name='y')
    return compile_precision_graph(Graph([header], [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def discrete_packet_words(rows, columns=0, samples=1):
    """64-byte RNG header, optional F32 weights, INT64 outputs and scratch."""
    if (any(type(n) is not int for n in (rows, columns, samples)) or rows < 1 or columns < 0 or samples < 1 or
            columns == 0 and samples != 1):
        raise ValueError('Invalid discrete RNG packet dimensions')
    words = max(18, 16+rows*columns, rows*columns+2*rows*samples+2)
    words = (words+1)//2*2
    if words > MAX_ELEMENTS:
        raise ValueError('Discrete RNG packet exceeds 65536 words')
    return words


def compile_discrete_packet(rows, columns=0, samples=1, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """Byte-DMA packet packing; seed, bounds and weights are not converted."""
    import numpy as np
    from . import Graph, input, op, constant
    words = discrete_packet_words(rows, columns, samples)
    header = input('header', shape=(1, 64), dtype='u8')
    inputs = [header]
    if columns:
        inputs.append(input('weights', shape=(1, rows*columns*4), dtype='u8'))
    parts = list(inputs)
    padding = words*4-64-rows*columns*4
    if padding:
        parts.append(constant(np.zeros((1, padding), dtype=np.uint8)))
    y = op('Concat', *parts, axis=1, _shape=(1, words*4), _dtype='u8', _name='y')
    return compile_precision_graph(Graph(inputs, [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_mask_packet(count, kind, code, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """Pack mixed storage widths with byte DMA; no numeric conversion."""
    import numpy as np
    from . import Graph, input, op, constant
    sizes, _, words = mask_packet_spec(count, kind, code)
    inputs = [input(f'x{i}', shape=(1, size), dtype='u8') for i, size in enumerate(sizes)]
    header = constant(np.array([count, kind, code, 0], dtype=np.float32).view(np.uint8).reshape(1, 16))
    parts = [header, *inputs]
    padding = words*4-16-sum(sizes)
    if padding:
        parts.append(constant(np.zeros((1, padding), dtype=np.uint8)))
    y = op('Concat', *parts, axis=1, _shape=(1, words*4), _dtype='u8', _name='y')
    return compile_precision_graph(Graph(inputs, [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_classification_pack(rows, columns, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """Byte DMA packs logits, INT64 target and INT64 ignore_index without casts."""
    from . import Graph, input, op
    if (type(rows) is not int or type(columns) is not int or min(rows, columns) < 1 or
            rows*(columns+4) > MAX_ELEMENTS):
        raise ValueError('NLL packet exceeds the bounded row carrier')
    x = input('x', shape=(rows, columns, 4), dtype='u8')
    target = input('target', shape=(rows, 2, 4), dtype='u8')
    ignore = input('ignore', shape=(rows, 2, 4), dtype='u8')
    y = op('Concat', x, target, ignore, axis=1, _shape=(rows, columns+4, 4), _dtype='u8', _name='y')
    return compile_precision_graph(Graph([x, target, ignore], [y]), libraries=libraries,
        native_dir=native_dir, timeout_ms=timeout_ms, dma_only=True)


def compile_rmsnorm(rows, kernel_elf, *, native_dir=None, timeout_ms=20_000, libraries=None, parallel=True):
    """Complete-row FP32 RMSNorm in the compiler's unmodified Power schedule.

    DefaultHW distributes >=8 rows across four tiles/two SHAVE tasks each.
    Small inputs retain ReferenceSW, because HW splits a single row's width.
    Gamma is pre-broadcast. Generic compute/storage limits are unchanged.
    """
    from . import Graph, input, op
    if (type(parallel) is not bool or type(rows) is not int or not 1 <= rows <= (512 if parallel else 256)
        or not isinstance(kernel_elf, bytes)):
        raise ValueError('RMSNorm requires 1..512 complete rows (serial <=256) and linked kernel bytes')
    distributed=parallel and rows>=8
    shape=(rows,576);x=input('x',shape=shape,dtype='f16');gamma=input('gamma',shape=shape,dtype='f16')
    y=op('Power',x,gamma,_shape=shape,_dtype='f16',_name='y')
    program=compile_precision_graph(Graph([x,gamma],[y],name='rmsnorm_fp32'),
        libraries=libraries,native_dir=native_dir,timeout_ms=timeout_ms,reference_sw=not distributed)
    elf=Elf(program.graph_blob);evidence=inspect(program.graph_blob,memref_count=3)
    if (any(s['size'] and s['name'].startswith('task.dpu.') for s in elf.sections)
        or len(evidence['ranges'])!=(8 if distributed else 1)
        or len(evidence['invocations'])!=(8 if distributed else 1)):
        raise ValueError('RMSNorm requires the verified complete-row SHAVE schedule and no DPU')
    total_rows=0;output_spans={tile:[] for tile in range(4)}
    for task in evidence['invocations']:
        refs=task.get('memrefs',[])
        if len(refs)!=3:raise ValueError('RMSNorm requires three tensor memrefs')
        count=refs[0]['elements']//576
        if not 1<=count<=(64 if distributed else 256):raise ValueError('RMSNorm row shard is out of bounds')
        dims=(1,1,576,1) if rows==1 else (576,count,1,1)
        strides=(16,16,16,16*576) if rows==1 else (16,16*576,16*576*count,16*576*count)
        for ref in refs:
            if (ref['dims']!=dims or ref['order']!='0x4321' or ref['strides_bits']!=strides):
                raise ValueError('RMSNorm carrier splits a row or has an unsupported stride')
        tile=task['tile'];address,span=_dense_float(refs[-1],tile,elf,'f16')
        if any(max(address,a)<min(address+span,a+s) for a,s in output_spans[tile]):
            raise ValueError('RMSNorm output row shards overlap')
        output_spans[tile].append((address,span));total_rows+=count
    if total_rows!=rows or (distributed and any(len(spans)!=2 for spans in output_spans.values())):
        raise ValueError('RMSNorm row schedule replicates or omits rows')
    info=program.ir_provenance
    blob,report=patch_binary_carrier(program.graph_blob,kernel_elf,vendor_id=info.vendor_id,
        device_id=info.device_id,compiler_version=info.compiler_version,dtype='f16')
    report.update(operation='rmsnorm_fp32',shape=list(shape),width=576,epsilon=1e-5,
        accumulation='fp32_neumaier',rounding='fp32_normalize_to_half_then_half_gamma_product',
        scheduling='four_tile_complete_rows' if distributed else 'single_tile_complete_rows')
    return replace(program,graph_blob=blob,patch_reports=(json.dumps(report).encode(),))


def compile_fused_rmsnorm(graph,xml,weights,kernel_elf,*,libraries,timeout_ms=20_000,whole_layer=False,select_kernel_elf=None):
    """Patch one segment carrier or two whole-layer RMSNorm carriers.

    Complete rows may be replicated per DPU tile. No buffer, DMA, barrier or
    task field is rewritten; only the fingerprinted Power code window changes.
    """
    from . import Program
    from .ir import SerializedIR
    if type(whole_layer) is not bool:raise TypeError('whole_layer must be bool')
    if select_kernel_elf is not None and not isinstance(select_kernel_elf,bytes):
        raise TypeError('select_kernel_elf must be linked ELF bytes')
    expected=['rmsnorm_carrier','rmsnorm_ffn'] if whole_layer else ['rmsnorm_carrier']
    powers=[n for n in graph.nodes if n.op=='Power']
    if sorted(n.name or '' for n in powers)!=expected:
        raise ValueError('Requires exactly the named RMSNorm carriers')
    for node in powers:
        if (len(node.outputs)!=1 or node.outputs[0].dtype!='f16' or len(node.inputs)!=2
            or any(v.dtype!='f16' or v.shape!=node.outputs[0].shape for v in node.inputs)
            or len(node.outputs[0].shape)!=2 or node.outputs[0].shape[1]!=576
            or not 2<=node.outputs[0].shape[0]<=512 or node.outputs[0].shape!=powers[0].outputs[0].shape):
            raise ValueError('Requires matching complete-row FP16 RMSNorm carriers')
    actual=[n for n in ET.fromstring(xml).findall('./layers/layer') if n.get('type')=='Power']
    if sorted(n.get('name','') for n in actual)!=expected:
        raise ValueError('Serialized graph must contain exactly the named RMSNorm carriers')
    for node in actual:
        ports=node.findall('./input/port')+node.findall('./output/port')
        if len(ports)!=3 or any(p.get('precision')!='FP16' or tuple(int(d.text) for d in p.findall('dim'))!=powers[0].outputs[0].shape for p in ports):
            raise ValueError('Serialized RMSNorm carrier shape or precision differs from the graph')
    result=libraries.compile_ir(xml,weights,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=timeout_ms)
    if (result.vendor_id,result.device_id,tuple(result.compiler_version))!=(0x8086,0x643e,(8,3)):
        raise ValueError('Unsupported fused RMSNorm device/compiler profile')
    elf=Elf(result.graph_blob)
    if not any(s['size'] and s['name'].startswith('task.dpu.') for s in elf.sections):
        raise ValueError('Fused RMSNorm requires DPU consumers')
    tasks=inspect(result.graph_blob,memref_count=3,kernel_sha256=POWER_TEXT_SHA256)['invocations']
    if not tasks:raise ValueError('Fused RMSNorm carrier was removed by the compiler')
    for task in tasks:
        if len(task.get('memrefs',[]))!=3:raise ValueError('Fused RMSNorm requires three static memrefs')
        for ref in task['memrefs']:
            count=ref['elements']//576
            rows=(ref['order']=='0x4321' and ref['dims']==(576,count,1,1)
                and ref['strides_bits']==(16,16*576,16*576*count,16*576*count))
            # Whole-layer compilation also emits complete rows with a leading
            # singleton axis. The same kernel skips that axis.
            singleton=(whole_layer and ref['order']=='0x2431' and ref['dims']==(1,576,count,1)
                and ref['strides_bits']==(16,16,16*576,16*576*count))
            if not 1<=count<=powers[0].outputs[0].shape[0] or ref['rank']!=4 or not (rows or singleton):
                raise ValueError('Fused RMSNorm carrier splits a row or changes its layout')
    blob,report=_patch_carrier(result.graph_blob,kernel_elf,vendor_id=result.vendor_id,device_id=result.device_id,
        compiler_version=result.compiler_version,dtype='f16',carrier='Power',input_count=2,select_kernel=True)
    report.update(operation='fused_rmsnorm_fp32',shape=list(powers[0].outputs[0].shape),carriers=expected,
        accumulation='fp32_neumaier',rounding='fp32_normalize_to_half_then_half_gamma_product',
        scheduling='compiler_complete_rows_may_replicate_per_dpu_tile')
    reports=[json.dumps(report).encode()]
    if select_kernel_elf is not None:
        blob,selection=patch_kv_select(blob,select_kernel_elf,vendor_id=result.vendor_id,device_id=result.device_id,
            compiler_version=result.compiler_version)
        selection['operation']='select_fp16_copy';reports.append(json.dumps(selection).encode())
    return Program(blob,graph,SerializedIR(xml,weights,()),result,tuple(reports),libraries,timeout_ms)


def compile_square_gemm(size, kernel_elf, *, native_dir=None, timeout_ms=20_000, libraries=None):
    """Experimental direct FP32 GEMM for complete 2..64 square matrices.

    ReferenceSW preserves whole Power operands; default HW mode splits them.
    The replacement kernel consumes A/B directly, with no expanded products.
    The existing Power fingerprint and CMX checks still apply unchanged.
    """
    from . import Graph, input, op
    if type(size) is not int or not 2 <= size <= 64:
        raise ValueError("Direct square GEMM requires an integer size in 2..64")
    if not isinstance(kernel_elf, bytes):
        raise TypeError("kernel_elf must be linked ELF bytes")
    x = input("x", shape=(size, size), dtype="f32")
    b = input("b", shape=(size, size), dtype="f32")
    y = op("Power", x, b, _shape=x.shape, _dtype="f32", _name="y")
    program = compile_precision_graph(Graph([x, b], [y], name="npu4000_square_gemm"),
        native_dir=native_dir, timeout_ms=timeout_ms, libraries=libraries, reference_sw=True)
    evidence = inspect(program.graph_blob, memref_count=3)
    if len(evidence["ranges"]) != 1 or len(evidence["invocations"]) != 1:
        raise ValueError("GEMM requires exactly one complete SHAVE invocation")
    for ref in evidence["invocations"][0].get("memrefs", []):
        if (ref["dims"] != (size, size, 1, 1) or ref["order"] != "0x4321" or
            ref["strides_bits"] != (32, 32*size, 32*size*size, 32*size*size)):
            raise ValueError("GEMM carrier does not bind complete row-major matrices")
    info = program.ir_provenance
    blob, report = patch_binary_carrier(program.graph_blob, kernel_elf,
        vendor_id=info.vendor_id, device_id=info.device_id,
        compiler_version=info.compiler_version, dtype="f32")
    report.update(operation="square_gemm", size=size, shape=[size, size],
                  compilation_mode="ReferenceSW", accumulation="fp32_neumaier_separate_mul_add")
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def _gemm_dma(elf):
    dma = []
    for section in elf.sections:
        if not section['name'].startswith('task.dma.'):
            continue
        if section['size'] % 224:
            raise ValueError("Unsupported parallel GEMM DMA descriptor size")
        raw = elf.content(section)
        for offset in range(0, len(raw), 224):
            desc = bounded(raw, offset, 224)
            src, dst = struct.unpack_from('<2Q', desc, 40)
            dma.append(dict(section=section['name'], offset=offset,
                cfg=struct.unpack_from('<Q', desc, 16)[0], width=struct.unpack_from('<2I', desc, 24),
                src=src, dst=dst, masks=struct.unpack_from('<4Q', desc, 64), raw=desc,
                file_offset=section['offset']+offset))
    return dma


def patch_parallel_square_gemm(graph_blob, kernel_elf, *, size, vendor_id, device_id, compiler_version):
    """Allow two row tasks to read B's full CMX upload after strict DMA checks.

    The last input DMA is forced in-order, so its completion barrier proves
    both full inputs are ready. No MemRef, DMA extent, address or barrier is
    expanded/rewritten; only the kernel and one ordering bit may change.
    """
    if type(size) is not int or not 2 <= size <= 64:
        raise ValueError("Parallel square GEMM requires an integer size in 2..64")
    # Reuse every existing profile/range/code/CMX/alias check before extra access.
    patched, report = patch_binary_carrier(graph_blob, kernel_elf, dtype="f32",
        vendor_id=vendor_id, device_id=device_id, compiler_version=compiler_version)
    elf = Elf(graph_blob)
    evidence = inspect(graph_blob, memref_count=3)
    tasks = evidence['invocations']
    if len(tasks) != 2 or any(t['tile'] != 0 or t['section'] != 'task.shave.invocation.0.0' for t in tasks):
        raise ValueError("Parallel GEMM requires two row tasks on tile zero")
    tasks = sorted(tasks, key=lambda t: int(t['memrefs'][0]['unrelocated_data'], 16))
    bases = [int(r['unrelocated_data'], 16) for r in tasks[0]['memrefs']]
    if bases[0] != 0x40218000:
        raise ValueError("Parallel GEMM requires the verified first A CMX address")
    span = size*size*4
    row = 0
    for task in tasks:
        if len(task['memrefs'][0]['dims']) != 4:
            raise ValueError("Parallel GEMM requires rank-four physical MemRefs")
        rows = task['memrefs'][0]['dims'][1]
        if not 0 < rows < size:
            raise ValueError("Parallel GEMM has an invalid row partition")
        for base, ref in zip(bases, task['memrefs']):
            if (ref['dims'] != (size, rows, 1, 1) or ref['order'] != '0x4321' or
                ref['strides_bits'] != (32, 32*size, 32*size*rows, 32*size*rows) or
                int(ref['unrelocated_data'], 16) != base+row*size*4):
                raise ValueError("Parallel GEMM rows are not complete/disjoint row-major partitions")
        row += rows
    if row != size:
        raise ValueError("Parallel GEMM does not cover every output row")
    # Validate whole A/B/C spans as well as the individual compiler MemRefs.
    full_spans = []
    for base, ref in zip(bases, tasks[0]['memrefs']):
        full = {**ref, 'dims': (size, size, 1, 1), 'elements': size*size,
                'unrelocated_data': hex(base)}
        full_spans.append(_dense_float(full, 0, elf, 'f32'))
    for i, (base, count) in enumerate(full_spans):
        if any(max(base, other) < min(base+count, other+extent) for other, extent in full_spans[i+1:]):
            raise ValueError("Parallel GEMM full matrices overlap")

    # NPU40XX DMA ABI: 224 bytes; cfg@16, widths@24, src@40, dst@48,
    # producer/wait low/high masks@64. Support only observed 1D byte copies.
    dma = _gemm_dma(elf)
    inputs = []
    for index in range(2):
        matches = [d for d in dma if any(r['target_section'] == d['section'] and
            r['target_offset'] == d['offset']+40 and r['symbol_section'] == f'io.NetworkInput.{index}'
            for r in elf.relocations)]
        if len(matches) != 1:
            raise ValueError("Parallel GEMM needs one complete DMA per input")
        d = matches[0]
        reloc = elf.relocation(d['section'], d['offset']+40)
        if (d['section'] != 'task.dma.0.0' or d['width'] != (span, span) or d['dst'] != bases[index] or
            d['cfg'] & ~(1 << 28) != 0x10000affff000 or d['raw'][32:40] != bytes(8) or
            d['raw'][96:180] != bytes(84) or d['src'] != 0 or
            reloc['type'] != 0 or reloc['symbol_value'] != 0 or reloc['addend'] != 0 or
            reloc['symbol_size'] != span):
            raise ValueError("Parallel GEMM input DMA lacks a full unconverted matrix")
        inputs.append(d)
    first, last = inputs
    link = elf.relocation(first['section'], first['offset'])
    if (link['type'] != 0 or link['symbol_section'] != last['section'] or
        link['symbol_value']+link['addend'] != last['offset']):
        raise ValueError("Parallel GEMM input DMA ordering chain differs")
    ready, wait, ready_hi, wait_hi = last['masks']
    if not ready or ready & (ready-1) or ready_hi or wait_hi or wait:
        raise ValueError("Parallel GEMM needs the final input completion barrier")
    done = None
    for task in tasks:
        raw = elf.content(elf.named[task['section']])
        hi, lo, post_hi, post_lo = struct.unpack_from('<4Q', raw, task['index']*96+32)
        if hi or lo != ready or post_hi or not post_lo or (done is not None and post_lo != done):
            raise ValueError("Parallel GEMM SHAVE completion dependencies differ")
        done = post_lo
    outputs = [d for d in dma if any(r['target_section'] == d['section'] and
        r['target_offset'] == d['offset']+48 and r['symbol_section'] == 'io.NetworkOutput.0'
        for r in elf.relocations)]
    if len(outputs) != 1:
        raise ValueError("Parallel GEMM needs one full output DMA")
    d = outputs[0]
    out_reloc = elf.relocation(d['section'], d['offset']+48)
    if (d['src'] != bases[2] or d['width'] != (span, span) or d['masks'][1] != done or d['masks'][3] or
        d['cfg'] != 0x10000bffff000 or d['raw'][32:40] != bytes(8) or d['raw'][96:180] != bytes(84) or
        out_reloc['type'] != 0 or out_reloc['symbol_value'] or out_reloc['addend'] or out_reloc['symbol_size'] != span):
        raise ValueError("Parallel GEMM output DMA lacks a complete synchronized matrix")
    # Other descriptor copies may load task metadata, but must not write inputs/C.
    for d in dma:
        if d in inputs or d in outputs:
            continue
        if any(max(d['dst'], base) < min(d['dst']+d['width'][1], base+span) for base in bases):
            raise ValueError("Other DMA writes a parallel GEMM matrix")
    mutable = bytearray(patched)
    order_offset = last['file_offset']+16
    struct.pack_into('<Q', mutable, order_offset, last['cfg'] | (1 << 28))
    start, end = report['code_offset'], report['code_offset']+report['code_bytes']
    if any(a != b and not (start <= i < end or i == order_offset+3)
           for i, (a, b) in enumerate(zip(graph_blob, mutable))):
        raise ValueError("Parallel GEMM changed bytes outside code/ordering bit")
    report.update(operation='parallel_square_gemm', size=size, shape=[size, size],
                  compilation_mode='DefaultHW_one_tile', full_matrix_spans=full_spans,
                  dma_input_order_forced=True, dma_order_offset=order_offset,
                  input_ready_barrier=ready, output_ready_barrier=done,
                  row_partitions=[t['memrefs'][0]['dims'][1] for t in tasks],
                  patched_sha256=hashlib.sha256(mutable).hexdigest(),
                  changed_bytes=sum(a != b for a, b in zip(graph_blob, mutable)))
    return bytes(mutable), report


def compile_parallel_square_gemm(size, kernel_elf, *, native_dir=None, timeout_ms=20_000, libraries=None):
    from . import Graph, input, op
    if type(size) is not int or not 2 <= size <= 64:
        raise ValueError("Parallel square GEMM requires an integer size in 2..64")
    x = input('x', shape=(size, size), dtype='f32')
    b = input('b', shape=(size, size), dtype='f32')
    y = op('Power', x, b, _shape=x.shape, _dtype='f32', _name='y')
    program = compile_precision_graph(Graph([x, b], [y], name='npu4000_parallel_square_gemm'),
        native_dir=native_dir, timeout_ms=timeout_ms, libraries=libraries, single_tile=True)
    info = program.ir_provenance
    blob, report = patch_parallel_square_gemm(program.graph_blob, kernel_elf, size=size,
        vendor_id=info.vendor_id, device_id=info.device_id, compiler_version=info.compiler_version)
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def patch_four_tile_square_gemm(graph_blob, kernel_elf, *, size, vendor_id, device_id, compiler_version):
    """Four complete A/B replicas in four tile workspaces; no remote CMX reads.

    Each tile computes disjoint rows and zeros the rest. A separate existing
    device reduction assembles C. Preserve all graph layouts/extents/barriers.
    """
    if type(size) is not int or size not in (8, 16, 32, 64):
        raise ValueError("Four-tile GEMM requires size 8, 16, 32 or 64")
    patched, report = _patch_four_tile_buffers(graph_blob, kernel_elf, count=size*size,
        vendor_id=vendor_id, device_id=device_id, compiler_version=compiler_version)
    report.update(operation='four_tile_square_gemm_partials', size=size)
    return patched, report


def _patch_four_tile_buffers(graph_blob, kernel_elf, *, count, vendor_id, device_id, compiler_version, replicas=4):
    if type(count) is not int or not 64 <= count <= 16384 or count % 64:
        raise ValueError("Four-tile GEMM buffer requires 64-aligned capacity in 64..16384")
    if type(replicas) is not int or replicas not in (4,8):
        raise ValueError("GEMM requires four or eight complete replicas")
    patched, report = patch_binary_carrier(graph_blob, kernel_elf, dtype='f32',
        vendor_id=vendor_id, device_id=device_id, compiler_version=compiler_version)
    elf = Elf(graph_blob)
    tasks = inspect(graph_blob, memref_count=3)['invocations']
    lanes = replicas//4
    if len(tasks) != replicas or sorted(t['tile'] for t in tasks) != [tile for tile in range(4) for _ in range(lanes)]:
        raise ValueError("Four-tile GEMM task replication differs")
    tasks.sort(key=lambda t: (t['tile'],int(t['memrefs'][0]['unrelocated_data'],16)))
    span = count*4
    bases, ready, done = {}, None, None
    for replica,task in enumerate(tasks):
        tile = task['tile']
        refs = task['memrefs']
        addresses = [int(r['unrelocated_data'], 16) for r in refs]
        base = 0x40018000+(1 << (21+tile))
        if addresses[0] != base+(replica%lanes)*span:
            raise ValueError("Four-tile GEMM first A address differs")
        for ref in refs:
            rows = 2 if replicas==4 else 8
            if (ref['dims'] != (count//rows, rows, 1, 1) or ref['order'] != '0x4321' or
                ref['strides_bits'] != (32, 32*count//rows, 32*count, 32*count)):
                raise ValueError("Four-tile GEMM lacks a complete dense matrix")
        if replicas==8 and addresses != [base+(i*lanes+replica%lanes)*span for i in range(3)]:
            raise ValueError("Eight-task GEMM replica addresses differ")
        if max(addresses[0], addresses[1]) < min(addresses[0]+span, addresses[1]+span):
            raise ValueError("Four-tile GEMM inputs overlap")
        bases[replica] = addresses
        raw = elf.content(elf.named[task['section']])
        hi, lo, post_hi, post_lo = struct.unpack_from('<4Q', raw, task['index']*96+32)
        if (hi or post_hi or not lo or lo & (lo-1) or not post_lo or post_lo & (post_lo-1) or
            (ready is not None and (lo != ready or post_lo != done))):
            raise ValueError("Four-tile GEMM task dependencies differ")
        ready, done = lo, post_lo
    dma = _gemm_dma(elf)
    inputs, outputs = [], []
    for tile in range(4):
        for index in range(3):
            source = index < 2
            field = 40 if source else 48
            name = f'io.NetworkInput.{index}' if source else 'io.NetworkOutput.0'
            matches = []
            for d in dma:
                rels = [r for r in elf.relocations if r['target_section'] == d['section'] and
                    r['target_offset'] == d['offset']+field and r['symbol_section'] == name and r['addend'] == tile*lanes*span]
                if rels:
                    matches.append((d, elf.relocation(d['section'], d['offset']+field)))
            if len(matches) != 1:
                raise ValueError("Four-tile GEMM replica DMA is missing/ambiguous")
            d, reloc = matches[0]
            cfg = 0x10000affff000
            if (d['width'] != (lanes*span, lanes*span) or (d['dst'] if source else d['src']) != bases[tile*lanes][index] or
                (d['src'] if source else d['dst']) != 0 or
                d['cfg'] & ~(1 << 28) != cfg or
                d['raw'][32:40] != bytes(8) or d['raw'][96:180] != bytes(84) or
                reloc['type'] != 0 or reloc['symbol_value'] or reloc['symbol_size'] != replicas*span):
                raise ValueError("Four-tile GEMM DMA is not a complete unconverted replica")
            (inputs if source else outputs).append(d)
    # Both input queues post the same original ready barrier. Force each group
    # in-order so the last completion proves all earlier uploads in that queue.
    groups = {}
    for d in inputs:
        groups.setdefault(d['section'], []).append(d)
    if len(groups) != 2:
        raise ValueError("Four-tile GEMM input queue grouping differs")
    for group in groups.values():
        group.sort(key=lambda d: d['offset'])
        if len(group) != 4 or group[-1]['masks'] != (ready, 0, 0, 0):
            raise ValueError("Four-tile GEMM input queue lacks its final completion")
        for previous, current in zip(group, group[1:]):
            link = elf.relocation(previous['section'], previous['offset'])
            if (link['type'] != 0 or link['symbol_section'] != current['section'] or
                link['symbol_value']+link['addend'] != current['offset'] or previous['masks'][0] or previous['masks'][2]):
                raise ValueError("Four-tile GEMM input ordering chain differs")
    output_groups = {}
    for d in outputs:
        output_groups.setdefault(d['section'], []).append(d)
    if len(output_groups) != 2:
        raise ValueError("Four-tile GEMM output queue grouping differs")
    for group in output_groups.values():
        group.sort(key=lambda d: d['offset'])
        if len(group) != 2 or group[0]['masks'][1] != done or group[0]['masks'][3]:
            raise ValueError("Four-tile GEMM output queue lacks its SHAVE completion wait")
        for previous,current in zip(group,group[1:]):
            link = elf.relocation(previous['section'], previous['offset'])
            if (link['type'] != 0 or link['symbol_section'] != current['section'] or
                link['symbol_value']+link['addend'] != current['offset']):
                raise ValueError("Four-tile GEMM output ordering chain differs")
    for d in dma:
        if d in inputs or d in outputs:
            continue
        if any(max(d['dst'], base) < min(d['dst']+d['width'][1], base+span)
               for addresses in bases.values() for base in addresses):
            raise ValueError("Other DMA overwrites a four-tile GEMM matrix")
    mutable = bytearray(patched)
    order_offsets = []
    for d in inputs + outputs:
        offset = d['file_offset']+16
        struct.pack_into('<Q', mutable, offset, d['cfg'] | (1 << 28))
        order_offsets.append(offset)
    allowed = {offset+3 for offset in order_offsets}
    start, end = report['code_offset'], report['code_offset']+report['code_bytes']
    if any(a != b and not (start <= i < end or i in allowed) for i, (a,b) in enumerate(zip(graph_blob, mutable))):
        raise ValueError("Four-tile GEMM changed bytes outside code/ordering bits")
    report.update(operation='four_tile_gemm_buffers', capacity=count, replicas=replicas,
        dma_order_offsets=order_offsets, input_ready_barrier=ready, output_ready_barrier=done,
        full_matrix_addresses=bases, patched_sha256=hashlib.sha256(mutable).hexdigest(),
        changed_bytes=sum(a != b for a,b in zip(graph_blob, mutable)))
    return bytes(mutable), report


def rect_gemm_capacity(shape):
    """Bounded full-K tile; four scalar shape arguments prefix the first input."""
    if (len(shape) != 3 or any(type(v) is not int for v in shape) or
        not 1 <= shape[0] <= 64 or not 1 <= shape[1] <= 256 or not 1 <= shape[2] <= 64):
        raise ValueError("Rectangular GEMM requires M/N in 1..64 and K in 1..256")
    m, k, n = shape
    count = (max(4+m*k, k*n, m*n)+63)//64*64
    if count > 16384:
        raise ValueError("Rectangular GEMM packed capacity exceeds 16384")
    return count


def compile_rect_gemm(shape, kernel_elf, *, libraries=None, native_dir=None, timeout_ms=20_000, replicas=4):
    """Experimental packed FP32 graph; returns four or eight partial-C replicas."""
    from . import Graph, input, op
    count = rect_gemm_capacity(shape)
    if type(replicas) is not int or replicas not in (4,8):
        raise ValueError("GEMM requires four or eight complete replicas")
    rows = 2 if replicas==4 else 8
    packed_shape = (replicas, rows, count//rows)
    x = input('x', shape=packed_shape, dtype='f32')
    b = input('b', shape=packed_shape, dtype='f32')
    y = op('Power', x, b, _shape=packed_shape, _dtype='f32', _name='y')
    program = compile_precision_graph(Graph([x, b], [y], name='rect_gemm'),
        libraries=libraries, native_dir=native_dir, timeout_ms=timeout_ms)
    info = program.ir_provenance
    blob, report = _patch_four_tile_buffers(program.graph_blob, kernel_elf, count=count, replicas=replicas,
        vendor_id=info.vendor_id, device_id=info.device_id, compiler_version=info.compiler_version)
    report.update(operation='rect_gemm_partials', shape=list(shape), header=[12345, *shape])
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def compile_four_tile_square_gemm(size, kernel_elf, *, libraries=None, native_dir=None, timeout_ms=20_000):
    from . import Graph, input, op
    if type(size) is not int or size not in (8, 16, 32, 64):
        raise ValueError("Four-tile GEMM requires size 8, 16, 32 or 64")
    shape = (4, 2, size*size//2)
    x = input('x', shape=shape, dtype='f32')
    b = input('b', shape=shape, dtype='f32')
    y = op('Power', x, b, _shape=shape, _dtype='f32', _name='y')
    program = compile_precision_graph(Graph([x, b], [y], name='four_tile_gemm'),
        libraries=libraries, native_dir=native_dir, timeout_ms=timeout_ms)
    info = program.ir_provenance
    blob, report = patch_four_tile_square_gemm(program.graph_blob, kernel_elf, size=size,
        vendor_id=info.vendor_id, device_id=info.device_id, compiler_version=info.compiler_version)
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def compile_precision_graph(graph, *, native_dir=None, timeout_ms=20_000, libraries=None, dma_only=False,
                            reference_sw=False, single_tile=False):
    """Driver compilation with precision preservation and strict FP32 checks.

    Pure FP32 arithmetic must contain SHAVE tasks, FP32 MemRefs, and no DPU
    tasks. Ordinary FP32 MatMul fails this contract: use Multiply+ReduceSum.
    Explicit Convert graphs may intentionally change precision.
    """
    from . import NativeLibraries, Program, serialize_ir
    native = libraries or NativeLibraries(native_dir)
    ir = serialize_ir(graph)
    xml = ET.fromstring(ir.xml)
    for layer in xml.findall("./layers/layer"):
        if layer.get("type") not in ("Parameter", "Result", "Const"):
            info = layer.find("rt_info")
            if info is None:
                info = ET.SubElement(layer, "rt_info")
            ET.SubElement(info, "attribute", name="DisablePrecisionConversion", version="0", value="dynamic:f16")
    ir = replace(ir, xml=ET.tostring(xml))
    flags = '--config EXECUTION_MODE_HINT="ACCURACY"'
    if reference_sw or single_tile:
        flags += ' NPU_TILES="1"'
    if reference_sw:
        flags += ' NPU_COMPILATION_MODE="ReferenceSW"'
    result = native.compile_ir(ir.xml, ir.weights, build_flags=flags,
                               timeout_ms=timeout_ms)
    if (result.vendor_id, result.device_id, tuple(result.compiler_version)) != (0x8086, 0x643E, (8, 3)):
        raise ValueError("Unsupported device/compiler profile")
    pure_fp32 = all(t.dtype == "f32" for t in (*graph.inputs, *graph.outputs))
    if dma_only:
        elf = Elf(result.graph_blob)
        if (any(s["size"] and s["name"].startswith(("task.dpu.", "task.shave.")) for s in elf.sections) or
            not any(s["size"] and s["name"].startswith("task.dma.") for s in elf.sections)):
            raise ValueError("Transport requires the verified DMA-only carrier")
    elif pure_fp32:
        elf = Elf(result.graph_blob)
        if any(s["size"] and s["name"].startswith("task.dpu.") for s in elf.sections):
            raise ValueError("FP32 graph contains DPU tasks; implicit precision lowering rejected")
        evidence = inspect(result.graph_blob)
        refs = [r for task in evidence["invocations"] for r in task.get("memrefs", [])]
        if not refs or any(r["dtype"] != 1 for r in refs):
            raise ValueError("FP32 graph lacks an all-FP32 SHAVE contract")
    return Program(result.graph_blob, graph, ir, result, (), native, timeout_ms)


def compile_cast(shape, *, dtype, kernel_elf, native_dir=None, timeout_ms=20_000, libraries=None):
    """Compile a subnormal-preserving SHAVE cast transport.

    f16->f32 returns ordinary f32. f32->f16 returns binary16 bit codes in
    uint32 lanes of an f32 carrier: decode low 16 bits, never numerically
    convert the carrier floats. torch_npu_minimal handles that unpacking.
    """
    from . import Graph, input, op
    if dtype not in ("f16", "f32"):
        raise ValueError("Cast input must be f16/f32")
    x = input("x", shape=shape, dtype=dtype)
    if not 1 <= len(x.shape) <= 4 or prod(x.shape) > MAX_ELEMENTS:
        raise ValueError("Cast requires 1-4 dimensions and at most 65536 elements")
    if not isinstance(kernel_elf, bytes):
        raise ValueError("Cast requires the subnormal-preserving linked kernel")
    if dtype == "f32":
        program = compile_unary(shape, kernel_elf, dtype="f32", native_dir=native_dir,
                                timeout_ms=timeout_ms, libraries=libraries)
        report = json.loads(program.patch_reports[0])
        report["output_encoding"] = "binary16-in-low-16-bits-of-u32"
        return replace(program, patch_reports=(json.dumps(report).encode(),))
    output_dtype = "f32" if dtype == "f16" else "f16"
    y = op("Convert", x, destination_type=output_dtype, _shape=x.shape, _dtype=output_dtype, _name="y")
    program = compile_precision_graph(Graph([x], [y]), native_dir=native_dir, timeout_ms=timeout_ms, libraries=libraries)
    if dtype == "f16":
        info = program.ir_provenance
        blob, report = _patch_carrier(program.graph_blob, kernel_elf, vendor_id=info.vendor_id,
            device_id=info.device_id, compiler_version=info.compiler_version, dtype=dtype,
            output_dtype=output_dtype, carrier="Convert", input_count=1)
        return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))
    return program


def compile_halfcode_slice(elements, *, libraries=None, native_dir=None, timeout_ms=20_000):
    """DMA compaction of exact binary16 codes stored in low halves of u32 lanes.

    Input is an internal (elements, 2) FP16 view of the FP32 cast allocation.
    No numeric conversion, host unpacking, or floating point computation.
    """
    import numpy as np
    from . import Graph, input, op, constant
    if not isinstance(elements, int) or not 1 <= elements <= MAX_ELEMENTS:
        raise ValueError("Halfcode transport requires 1-65536 codes")
    x = input("x", shape=(elements, 2), dtype="f16")
    begin, end, strides = [constant(np.array(v, dtype=np.int64))
                           for v in ([0, 0], [elements, 1], [1, 1])]
    y = op("StridedSlice", x, begin, end, strides, begin_mask=[0, 0], end_mask=[0, 0],
           new_axis_mask=[0, 0], shrink_axis_mask=[0, 0], ellipsis_mask=[0, 0],
           _shape=(elements, 1), _dtype="f16", _name="y")
    program = compile_precision_graph(Graph([x], [y]), libraries=libraries,
                                       native_dir=native_dir, timeout_ms=timeout_ms)
    elf = Elf(program.graph_blob)
    if (any(s["size"] and s["name"].startswith(("task.dpu.", "task.shave.")) for s in elf.sections) or
        not any(s["size"] and s["name"].startswith("task.dma.") for s in elf.sections)):
        raise ValueError("Halfcode compaction requires the verified DMA-only carrier")
    return program


def compile_reduce(shape, axes, kernel_elf, *, kind, libraries=None, native_dir=None, timeout_ms=20_000):
    """Patch a complete FP32 ReduceSum/ReduceMean, rejecting split reduction axes."""
    import numpy as np
    from . import Graph, input, op, constant
    if kind not in ("sum", "mean"):
        raise ValueError("Reduction kind must be sum/mean")
    x = input("x", shape=shape, dtype="f32")
    axes = tuple(axes)
    if (not 1 <= len(x.shape) <= 4 or prod(x.shape) > MAX_ELEMENTS or not axes or
        len(set(axes)) != len(axes) or any(not isinstance(a, int) or not 0 <= a < len(x.shape) for a in axes)):
        raise ValueError("Reduction requires rank 1-4, 1-65536 elements, and distinct valid axes")
    count = prod(x.shape[a] for a in axes)
    kwargs = dict(libraries=libraries, native_dir=native_dir, timeout_ms=timeout_ms)
    if count == 1:
        # The compiler removes a singleton reduction. Use the verified Abs ABI;
        # this kernel still performs +0 accumulation (including negative zero).
        return compile_unary(shape, kernel_elf, dtype="f32", **kwargs)
    outshape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
    carrier = "ReduceSum" if kind == "sum" else "ReduceMean"
    y = op(carrier, x, constant(np.array(axes, dtype=np.int64)), keep_dims=True,
           _shape=outshape, _dtype="f32", _name="y")
    program = compile_precision_graph(Graph([x], [y]), **kwargs)
    info = program.ir_provenance
    blob, report = _patch_carrier(program.graph_blob, kernel_elf, vendor_id=info.vendor_id,
        device_id=info.device_id, compiler_version=info.compiler_version, dtype="f32",
        carrier=carrier, input_count=1, reduction_count=count)
    report.update(shape=list(x.shape), axes=list(axes), reduction_count=count)
    return replace(program, graph_blob=blob, patch_reports=(json.dumps(report).encode(),))


def compile_broadcast(shape, target, *, dtype, libraries=None, native_dir=None, timeout_ms=20_000):
    """Materialize a static NumPy-style broadcast with NPU DMA, preserving bits."""
    import numpy as np
    from . import Graph, input, op, constant
    if dtype not in ("f16", "f32", "u8"):
        raise ValueError("Broadcast dtype must be f16/f32/u8")
    x = input("x", shape=shape, dtype=dtype)
    target = tuple(target)
    if (not 1 <= len(x.shape) <= len(target) <= 4 or prod(target) > MAX_ALLOCATION_BYTES//{'u8':1,'f16':2,'f32':4}[dtype] or
        any(not isinstance(d, int) or d < 1 for d in target) or
        any(a not in (1, b) for a, b in zip((1,) * (len(target) - len(x.shape)) + x.shape, target))):
        raise ValueError("Broadcast requires compatible rank 1-4 shapes with bounded output elements")
    if dtype == "f32":
        # FP32 Broadcast introduces compute tasks on this compiler. Replicate
        # pairs of 16-bit storage words instead, without any float conversion.
        x = input("x", shape=(*x.shape, 2), dtype="f16")
        target = (*target, 2)
    transport_dtype = x.dtype
    y = op("Broadcast", x, constant(np.array(target, dtype=np.int64)), mode="numpy",
           _shape=target, _dtype=transport_dtype, _name="y")
    return compile_precision_graph(Graph([x], [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_permute(shape, order, *, dtype, libraries=None, native_dir=None, timeout_ms=20_000):
    """Reorder complete dense allocations with DMA, including exact FP32 bits."""
    import numpy as np
    from . import Graph, input, op, constant
    if dtype not in ("f16", "f32", "u8"):
        raise ValueError("Permutation dtype must be f16/f32/u8")
    x = input("x", shape=shape, dtype=dtype)
    order = tuple(order)
    if (not 1 <= len(x.shape) <= 4 or prod(x.shape) > MAX_ALLOCATION_BYTES//{"u8":1,"f16":2,"f32":4}[dtype] or
        any(type(a) is not int for a in order) or sorted(order) != list(range(len(x.shape)))):
        raise ValueError("Permutation requires rank 1-4, bounded elements, and every axis exactly once")
    x = input("x", shape=(*x.shape, {"u8": 1, "f16": 2, "f32": 4}[dtype]), dtype="u8")
    order = (*order, len(order))  # Keep the bytes of each floating point value together.
    target = tuple(x.shape[a] for a in order)
    y = op("Transpose", x, constant(np.array(order, dtype=np.int64)),
           _shape=target, _dtype=x.dtype, _name="y")
    return compile_precision_graph(Graph([x], [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_slice(elements, start, count, *, dtype, libraries=None, native_dir=None, timeout_ms=20_000):
    """Extract a contiguous compute tile with exact byte DMA, without host copies."""
    import numpy as np
    from . import Graph, input, op, constant
    if (dtype not in ("f16", "f32", "u8") or any(type(v) is not int for v in (elements, start, count)) or
        not 1 <= elements <= MAX_ALLOCATION_BYTES//{'u8':1,'f16':2,'f32':4}[dtype] or not 1 <= count <= MAX_ELEMENTS or
        not 0 <= start <= elements-count):
        raise ValueError("Slice requires bounded storage and an in-range 1-65536-element tile")
    width = {"u8": 1, "f16": 2, "f32": 4}[dtype]
    x = input("x", shape=(elements, width), dtype="u8")
    begin, end, strides = [constant(np.array(v, dtype=np.int64))
                           for v in ([start, 0], [start+count, width], [1, 1])]
    y = op("StridedSlice", x, begin, end, strides, begin_mask=[0, 0], end_mask=[0, 0],
           new_axis_mask=[0, 0], shrink_axis_mask=[0, 0], ellipsis_mask=[0, 0],
           _shape=(count, width), _dtype="u8", _name="y")
    return compile_precision_graph(Graph([x], [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_gather(elements, indices, *, dtype, libraries=None, native_dir=None, timeout_ms=20_000):
    """Exact byte gather with a zero sentinel, for im2col/col2im transports."""
    import numpy as np
    from . import Graph, input, op, constant
    indices = np.asarray(indices)
    if (dtype not in ("f16", "f32", "u8") or type(elements) is not int or not 1 <= elements <= MAX_ELEMENTS or
        indices.dtype.kind not in "iu" or not 1 <= indices.ndim <= 3 or not 0 < indices.size <= MAX_ELEMENTS or
        np.any(indices < -1) or np.any(indices >= elements)):
        raise ValueError("Gather requires bounded FP16/FP32 storage and indices in [-1,elements)")
    width = {"u8": 1, "f16": 2, "f32": 4}[dtype]
    x = input("x", shape=(elements, width), dtype="u8")
    padded = op("Concat", x, constant(np.zeros((1, width), dtype=np.uint8)), axis=0,
                _shape=(elements+1, width), _dtype="u8")
    y = op("Gather", padded, constant(np.where(indices < 0, elements, indices).astype(np.int64)),
           constant(np.array(0, dtype=np.int64)), _shape=(*indices.shape, width), _dtype="u8", _name="y")
    return compile_precision_graph(Graph([x], [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_matrix_concat(rows, columns, *, dtype, libraries=None, native_dir=None, timeout_ms=20_000):
    """Assemble matrix row tiles with byte DMA; no numeric conversion."""
    from . import Graph, input, op
    rows = tuple(rows)
    if (dtype not in ("f16", "f32", "u8") or type(columns) is not int or columns < 1 or not 2 <= len(rows) <= 63 or
        any(type(r) is not int or r < 1 for r in rows) or sum(rows)*columns > MAX_ALLOCATION_BYTES//{'u8':1,'f16':2,'f32':4}[dtype]):
        raise ValueError("Matrix concat requires 2-63 positive row tiles and bounded output elements")
    width = {"u8": 1, "f16": 2, "f32": 4}[dtype]
    inputs = [input(f"x{i}", shape=(r, columns, width), dtype="u8") for i, r in enumerate(rows)]
    y = op("Concat", *inputs, axis=0, _shape=(sum(rows), columns, width), _dtype="u8", _name="y")
    return compile_precision_graph(Graph(inputs, [y]), libraries=libraries, native_dir=native_dir,
                                   timeout_ms=timeout_ms, dma_only=True)


def compile_concat_many(groups, *, dtype, libraries=None, native_dir=None, timeout_ms=20_000):
    """Independent byte-DMA concatenations in one multi-output graph."""
    from . import Graph, input, op
    groups=tuple(tuple(group) for group in groups)
    if (dtype not in ('f16','f32') or not 2<=len(groups) or sum(map(len,groups))+len(groups)>64
        or any(not 2<=len(group)<=63 or any(type(size) is not int or size<1 for size in group)
               or sum(group)>MAX_ALLOCATION_BYTES//(2 if dtype=='f16' else 4) for group in groups)):
        raise ValueError('Multi-concat requires >=2 outputs, 2-63 tiles each, <=64 total bindings and bounded outputs')
    width=2 if dtype=='f16' else 4;inputs=[];outputs=[]
    for i,group in enumerate(groups):
        tiles=[input(f'g{i}_x{j}',shape=(size,1,width),dtype='u8') for j,size in enumerate(group)]
        inputs.extend(tiles)
        outputs.append(op('Concat',*tiles,axis=0,_shape=(sum(group),1,width),_dtype='u8',_name=f'y{i}'))
    return compile_precision_graph(Graph(inputs,outputs),libraries=libraries,native_dir=native_dir,
                                   timeout_ms=timeout_ms,dma_only=True)


