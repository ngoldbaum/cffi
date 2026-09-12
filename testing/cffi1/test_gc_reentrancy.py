import subprocess
import sys

import pytest

import _cffi_backend as backend
from cffi import FFI
from testing.cffi1.test_recompiler import verify


pytestmark = pytest.mark.thread_unsafe(
    reason="Compiles an extension and runs it in subprocesses")


@pytest.fixture(scope="module")
def module_path():
    ffi = FFI()
    ffi.cdef('struct lazy_conv { char a; }; extern "Python" int callback(int);')
    verify(ffi, "gc_reentrancy", "struct lazy_conv { char a; };")
    return sys.modules["_CFFI_gc_reentrancy"].__file__


@pytest.mark.skipif(sys.implementation.name != "cpython" or
                    sys.version_info >= (3, 12),
                    reason="allocation-triggered GC only exists before 3.12")
def test_gc_finalizer_reentry(module_path):
    # A finalizer that uses the struct being completed and registers the
    # extern function being registered, at every allocation offset of those
    # operations.  Before the fix this produced "types are different" errors
    # and segfaults, which is how issue #271 presented.
    script = """
import gc, sys
from cffi._imp_emulation import load_dynamic
module = load_dynamic("_CFFI_gc_reentrancy", %r)
ffi, lib = module.ffi, module.lib
class Finalizer:
    def __del__(self):
        ffi.new("struct lazy_conv *")
        ffi.def_extern(name="callback")(lambda value: 22)
gc.disable()
gc.collect()
garbage = Finalizer()
garbage.me = garbage
del garbage
gc.set_threshold(gc.get_count()[0] + int(sys.argv[1]), 0, 0)
gc.enable()
ct = ffi.typeof("struct lazy_conv")
ffi.sizeof(ct)
ffi.def_extern(name="callback")(lambda value: 11)
p = ffi.new("struct lazy_conv *")
p.a = b"x"
assert ffi.typeof("struct lazy_conv") is ct
assert [n for n, f in ct.fields] == ["a"]
assert lib.callback(0) in (11, 22)
""" % module_path
    for offset in range(16):
        process = subprocess.run([sys.executable, "-c", script, str(offset)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=120)
        assert process.returncode == 0, (offset, process.returncode, process.stdout[-800:])


def test_complete_twice_keeps_layout():
    ct = backend.new_struct_type("struct twice")
    integer = backend.new_primitive_type("int")
    backend.complete_struct_or_union(ct, [("value", integer)])
    with pytest.raises(TypeError):
        backend.complete_struct_or_union(ct, [("other", integer)])
    assert [name for name, field in ct.fields] == ["value"]


def test_reentrant_explicit_completion():
    ct = backend.new_struct_type("struct reentrant")
    char = backend.new_primitive_type("char")
    integer = backend.new_primitive_type("int")

    class Reenter:
        def __index__(self):
            backend.complete_struct_or_union(ct, [("winner", integer)])
            return -1

    # The outer candidate has a different layout and packing flags; its
    # rejection must not clear or overwrite the nested call's result.
    fields = [("discarded", char), ("trigger", integer, Reenter())]
    with pytest.raises(TypeError):
        backend.complete_struct_or_union(ct, fields, None, -1, -1, 0, 1)
    assert [name for name, field in ct.fields] == ["winner"]
    assert backend.sizeof(ct) == backend.sizeof(integer)
