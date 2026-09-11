from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import gc
import subprocess
import sys
import time
import weakref

import pytest

from cffi import FFI
import _cffi_backend as backend
from testing.cffi1.test_recompiler import verify


pytestmark = pytest.mark.thread_unsafe(
    reason="Compiles extensions, then starts its own worker threads")


@pytest.fixture(scope="module", autouse=True)
def low_switch_interval():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


@pytest.mark.parametrize("other_thread", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_reentrant_explicit_completion(other_thread, fail):
    ct = backend.new_struct_type("struct reentrant")
    char = backend.new_primitive_type("char")
    integer = backend.new_primitive_type("int")

    def complete():
        backend.complete_struct_or_union(ct, [("winner", integer)])

    class Reenter:
        def __index__(self):
            if other_thread:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(complete).result(timeout=60)
            else:
                complete()
            if fail:
                raise ValueError("outer conversion failed")
            return -1

    # The outer candidate includes a different layout and packing flags.
    # Its failure must not clear or overwrite the nested call's result.
    fields = [("discarded", char), ("trigger", integer, Reenter())]
    error = ValueError if fail else TypeError
    with pytest.raises(error):
        backend.complete_struct_or_union(ct, fields, None, -1, -1, 0, 1)
    assert [name for name, field in ct.fields] == ["winner"]
    assert backend.sizeof(ct) == backend.sizeof(integer)
    assert backend.alignof(ct) == backend.alignof(integer)
    assert backend.new_function_type((ct,), ct).result is ct


def test_completion_failure_is_retryable():
    ct = backend.new_struct_type("struct retry")
    char = backend.new_primitive_type("char")
    integer = backend.new_primitive_type("int")
    flexible = backend.new_array_type(backend.new_pointer_type(char), None)
    with pytest.raises(KeyError):
        backend.complete_struct_or_union(
            ct, [("duplicate", integer), ("duplicate", flexible)])
    assert ct.fields is None
    backend.complete_struct_or_union(ct, [("value", integer)])
    assert backend.sizeof(ct) == backend.sizeof(integer)
    # The replacement layout must also be usable in a libffi signature.
    assert backend.new_function_type((ct,), ct).result is ct


@pytest.mark.parametrize("array", [False, True])
def test_concurrent_explicit_completion_readers(array):
    integer = backend.new_primitive_type("int")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i in range(32):
            ct = backend.new_struct_type("struct publication")
            ptr = backend.new_pointer_type(ct)
            barrier = Barrier(2, timeout=60)

            def complete():
                barrier.wait()
                backend.complete_struct_or_union(ct, [("value", integer)])

            def read():
                barrier.wait()
                for j in range(100):
                    try:
                        if array:
                            value = backend.newp(backend.new_array_type(ptr, 1))[0]
                        else:
                            value = backend.newp(ptr)
                    except (TypeError, ValueError) as error:
                        assert "unknown size" in str(error)
                    else:
                        assert value.value == 0
                        value.value = 42
                        assert value.value == 42

            a = pool.submit(complete)
            b = pool.submit(read)
            a.result(timeout=60)
            b.result(timeout=60)


@pytest.mark.parametrize("array", [False, True])
def test_lazy_completion_failure_is_retryable(array):
    ffi = FFI()
    declaration = "struct outer { struct { unsigned int value : %d; } inner%s; };"
    suffix = "[2]" if array else ""
    ffi.cdef(declaration % (100, suffix))
    verify(ffi, "lazy_completion_failure_is_retryable_%d" % array,
           declaration % (1, suffix))
    # Cached descriptors require the same layout validation on each attempt.
    for i in range(3):
        with pytest.raises(TypeError, match="exceeds the width"):
            ffi.sizeof("struct outer")


def _call_with_gc_reentry(function, argument, allocation_offset=0, observe=None):
    nested = []
    errors = []

    def callback(phase, info):
        if phase == "start":
            try:
                value = function(argument)
                nested.append(observe(value) if observe else value)
            except BaseException as error:
                errors.append(error)

    enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    gc.disable()
    gc.collect()
    try:
        gc.callbacks.append(callback)
        gc.set_threshold(gc.get_count()[0] + allocation_offset, 0, 0)
        # A zero threshold disables automatic GC, so keep it at least one.
        if gc.get_threshold()[0] == 0:
            gc.set_threshold(1, 0, 0)
        gc.enable()
        result = function(argument)
    finally:
        gc.disable()
        gc.callbacks.remove(callback)
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
    assert not errors
    assert nested, "the C call must actually trigger GC reentry"
    return result, nested


@pytest.mark.skipif(sys.implementation.name != "cpython" or
                    sys.version_info >= (3, 12),
                    reason="requires allocation-triggered GC in CPython <= 3.11")
@pytest.mark.parametrize("operation", ["struct", "enum", "function", "fields"])
def test_gc_reentrant_realization(operation):
    ffi = FFI()
    declaration = """
        struct item { int value; struct item *next; };
        enum mode { MODE = 42 };
        typedef struct item (*function_t)(struct item);
    """
    ffi.cdef(declaration)
    verify(ffi, "gc_reentrant_realization_" + operation, declaration)
    argument = {"struct": "struct item", "enum": "enum mode",
                "function": "function_t", "fields": "struct item"}[operation]
    if operation == "fields":
        argument = ffi.typeof(argument)
        result, nested = _call_with_gc_reentry(
            ffi.sizeof, argument, observe=lambda size: (size, argument.fields))
        assert all(size == result and fields == argument.fields
                   for size, fields in nested)
    else:
        result, nested = _call_with_gc_reentry(ffi.typeof, argument)
        assert all(value is result for value in nested)
        assert ffi.typeof(argument) is result
    ct = ffi.typeof("struct item")
    assert [name for name, field in ct.fields] == ["value", "next"]
    assert ct.fields[1][1].type.item is ct


@pytest.mark.skipif(sys.implementation.name != "cpython" or
                    sys.version_info >= (3, 12),
                    reason="requires allocation-triggered GC in CPython <= 3.11")
def test_gc_reentrant_unique_type():
    # A fresh base type avoids an existing pointer in the unique cache.
    ct = backend.new_struct_type("struct unique")
    result, nested = _call_with_gc_reentry(backend.new_pointer_type, ct, 1)
    assert all(value is result for value in nested)
    assert backend.new_pointer_type(ct) is result


@pytest.mark.skipif(sys.implementation.name != "cpython" or
                    sys.version_info >= (3, 12),
                    reason="requires allocation-triggered GC in CPython <= 3.11")
def test_gc_reentrant_first_extern_registration():
    ffi = FFI()
    ffi.cdef('extern "Python" int outer(void); extern "Python" int inner(void);')
    verify(ffi, "gc_reentrant_first_extern_registration", "")
    name = "_CFFI_gc_reentrant_first_extern_registration"
    path = sys.modules[name].__file__
    script = """
import gc, sys
from cffi._imp_emulation import load_dynamic
module = load_dynamic(%r, %r)
ffi, lib = module.ffi, module.lib
gc.disable()
def outer(): return 11
def inner(): return 22
reentered = []
def register(phase, info):
    if phase == "start":
        ffi.def_extern(name="inner")(inner)
        reentered.append(True)
gc.collect()
gc.callbacks.append(register)
gc.set_threshold(gc.get_count()[0] + int(sys.argv[1]), 1000000, 1000000)
gc.enable()
ffi.def_extern(name="outer")(outer)
gc.disable()
gc.callbacks.remove(register)
assert lib.outer() == 11
if reentered:
    assert lib.inner() == 22
    print("reentered")
""" % (name, path)
    # Each process starts with an empty callback registry.  Vary the GC
    # threshold to exercise allocations throughout the first registration.
    observed = False
    for offset in range(1, 25):
        process = subprocess.run(
            [sys.executable, "-c", script, str(offset)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=60)
        assert process.returncode == 0, process.stdout
        observed |= "reentered" in process.stdout
    assert observed


def test_concurrent_type_realization():
    ffi = FFI()
    declarations = []
    definitions = []
    for i in range(24):
        declarations.append("""
            enum mode%(i)d { MODE%(i)d = %(i)d };
            struct item%(i)d {
                struct { int value; };
                struct item%(i)d *next;
                enum mode%(i)d mode;
            };
            struct item%(i)d *echo%(i)d(struct item%(i)d *);
        """ % {"i": i})
        definitions.append("""
            struct item%(i)d *echo%(i)d(struct item%(i)d *p) { return p; }
        """ % {"i": i})
    ffi.cdef("\n".join(declarations))
    lib = verify(ffi, "concurrent_type_realization",
                 "\n".join(declarations + definitions))
    barrier = Barrier(4, timeout=60)

    def work(worker):
        types = []
        for i in range(24):
            barrier.wait()
            # Mix function signature realization with struct and anonymous
            # field realization, all using the same initially lazy type table.
            if worker % 2:
                getattr(lib, "echo%d" % i)
            ct = ffi.typeof("struct item%d" % i)
            # sizeof() completes the fields.  Reads during lazy completion
            # are exercised separately in test_concurrent_lazy_struct_size_readers.
            size = ffi.sizeof(ct)
            assert size > 0
            storage = bytearray(2 * size)
            array = ffi.from_buffer("struct item%d[]" % i, storage)
            ptr = ffi.cast("struct item%d *" % i, array)
            assert (ptr + 1) - ptr == 1
            assert ffi.addressof(array, 1) == ptr + 1
            assert len(ffi.buffer(ptr)) == size
            assert ffi.sizeof(ptr[0]) == size
            assert ffi.sizeof(ptr[0:2]) == 2 * size
            assert len(list(array)) == len(ffi.unpack(ptr, 2)) == 2
            p = ffi.new("struct item%d *" % i)
            p.value = worker
            p.next = p
            p.mode = i
            assert getattr(lib, "echo%d" % i)(p) == p
            assert ffi.addressof(lib, "echo%d" % i)(p) == p
            assert p.next.value == worker
            assert p.mode == i
            assert [name for name, field in ct.fields] == ["value", "next", "mode"]
            assert ffi.sizeof(ct) == ffi.sizeof(p[0])
            types.append(ct)
        return types

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(4)))
    for types in results[1:]:
        assert all(a is b for a, b in zip(results[0], types))


@pytest.mark.parametrize("operation", ["pointer_arithmetic", "pointer_conversion"])
def test_concurrent_lazy_struct_size_readers(operation):
    ffi = FFI()
    fields = "int a; double b;" if operation == "pointer_arithmetic" else "char a;"
    declaration = "struct lazy { %s };" % fields
    ffi.cdef(declaration)
    verify(ffi, "concurrent_lazy_struct_size_readers_" + operation, declaration)
    storage = ffi.new("char[128]")
    ptr = ffi.cast("struct lazy *", storage)
    slot = ffi.new("char **")
    started = Event()
    done = Event()

    def complete():
        try:
            assert started.wait(timeout=60)
            # Let the reader start without completing the lazy field list.
            time.sleep(0.05)
            ffi.new("struct lazy *")
        finally:
            done.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(complete)
        started.set()
        deadline = time.monotonic() + 60
        n = 0
        while not done.is_set() or n < 200:
            # Neither operation forces the struct's fields.  In particular,
            # do not call ffi.sizeof(ctype) here: that would complete them.
            if operation == "pointer_arithmetic":
                assert (ptr + 1) - ptr == 1
            else:
                # Implicit conversion to char * checks the pointee's size.
                slot[0] = ptr
            n += 1
            assert time.monotonic() < deadline
        future.result()


@pytest.mark.parametrize("collect", [False, True])
def test_concurrent_extern_python_redefinition(collect):
    ffi = FFI()
    ffi.cdef('extern "Python" int callback(int); int invoke(int);')
    lib = verify(ffi, "concurrent_extern_python_redefinition_%d" % collect, """
        static int callback(int);
        int invoke(int n) {
            int i;
            for (i = 0; i < n; i++) {
                int result = callback(i);
                if (result != i && result != i + 1)
                    return 0;
            }
            return 1;
        }
    """)
    ffi.def_extern(name="callback")(lambda value: value)
    assert lib.invoke(1) == 1
    workers = 5 if collect else 4
    barrier = Barrier(workers, timeout=60)
    callbacks = []

    def work(worker):
        barrier.wait()
        if worker == 0:
            for i in range(1000):
                offset = i % 2
                callback = lambda value, offset=offset: value + offset
                callbacks.append(weakref.ref(callback))
                ffi.def_extern(name="callback")(callback)
        elif worker == 4:
            for i in range(20):
                gc.collect()
        else:
            assert lib.invoke(1000) == 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(workers)))
    ffi.def_extern(name="callback")(lambda value: -1)
    assert lib.invoke(1) == 0
    if collect:
        gc.collect()
        gc.collect()
        assert all(ref() is None for ref in callbacks)



@pytest.mark.skipif(sys.platform == "win32", reason="uses pthreads")
def test_concurrent_callback_thread_cleanup():
    ffi = FFI()
    ffi.cdef("int run_callback(int (*)(int), int);")
    lib = verify(ffi, "concurrent_callback_thread_cleanup", """
        #include <pthread.h>
        struct callback_args {
            int (*callback)(int);
            int value;
            int result;
        };
        static void *call(void *ptr) {
            struct callback_args *args = (struct callback_args *)ptr;
            args->result = args->callback(args->value);
            return NULL;
        }
        int run_callback(int (*callback)(int), int value) {
            pthread_t thread;
            struct callback_args args = {callback, value, -1};
            if (pthread_create(&thread, NULL, call, &args) != 0)
                return -1;
            if (pthread_join(thread, NULL) != 0)
                return -1;
            return args.result;
        }
    """, extra_compile_args=["-pthread"], extra_link_args=["-pthread"])
    callback = ffi.callback("int(int)", lambda value: value * 2)
    barrier = Barrier(4, timeout=60)

    def work(worker):
        barrier.wait()
        for i in range(32):
            value = worker * 32 + i
            assert lib.run_callback(callback, value) == value * 2

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(4)))
