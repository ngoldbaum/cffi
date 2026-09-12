from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import gc
import sys
import time
import weakref

import pytest

from cffi import FFI
import _cffi_backend as backend
from testing.cffi1.test_recompiler import verify


pytestmark = pytest.mark.thread_unsafe(
    reason="Compiles extensions, then starts its own worker threads")

N_ITEMS = 24


@pytest.fixture(scope="module", autouse=True)
def low_switch_interval():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


@pytest.fixture(scope="module")
def shared():
    """One compiled module for the tests below; each test uses its own types."""
    ffi = FFI()
    declarations = []
    definitions = []
    for i in range(N_ITEMS):
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
    structs = "struct lazy_conv { char a; };"
    cdef = declarations + [structs, 'extern "Python" int callback(int); int invoke(int);']
    definitions.append(structs + """
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
    ffi.cdef("\n".join(cdef))
    lib = verify(ffi, "thread_safety", "\n".join(declarations + definitions))
    return ffi, lib


def test_concurrent_type_realization(shared):
    ffi, lib = shared
    barrier = Barrier(4, timeout=60)

    def work(worker):
        types = []
        for i in range(N_ITEMS):
            barrier.wait()
            # Mix function signature realization with struct and anonymous
            # field realization, all using the same initially lazy type table.
            if worker % 2:
                getattr(lib, "echo%d" % i)
            ct = ffi.typeof("struct item%d" % i)
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


def test_concurrent_lazy_struct_size_readers(shared):
    ffi, lib = shared
    storage = ffi.new("char[128]")
    ptr = ffi.cast("struct lazy_conv *", storage)
    slot = ffi.new("char **")
    started = Event()
    done = Event()

    def complete():
        try:
            assert started.wait(timeout=60)
            # Let the readers start without completing the lazy field list.
            time.sleep(0.05)
            ffi.new("struct lazy_conv *")
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
            assert (ptr + 1) - ptr == 1
            # Implicit conversion to char * checks the pointee's size.
            slot[0] = ptr
            n += 1
            assert time.monotonic() < deadline
        future.result()


def test_concurrent_extern_python_redefinition(shared):
    ffi, lib = shared
    ffi.def_extern(name="callback")(lambda value: value)
    assert lib.invoke(1) == 1
    barrier = Barrier(5, timeout=60)
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

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(work, range(5)))
    ffi.def_extern(name="callback")(lambda value: -1)
    assert lib.invoke(1) == 0
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


def test_reentrant_explicit_completion():
    ct = backend.new_struct_type("struct reentrant")
    char = backend.new_primitive_type("char")
    integer = backend.new_primitive_type("int")

    def complete():
        backend.complete_struct_or_union(ct, [("winner", integer)])

    class Reenter:
        def __index__(self):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(complete).result(timeout=60)
            return -1

    # The outer candidate includes a different layout and packing flags.
    # Its failure must not clear or overwrite the nested call's result.
    fields = [("discarded", char), ("trigger", integer, Reenter())]
    with pytest.raises(TypeError):
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


def test_concurrent_explicit_completion_readers():
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
                        if j % 2:
                            # keep the array alive: indexing does not
                            owner = backend.newp(backend.new_array_type(ptr, 1))
                            value = owner[0]
                        else:
                            owner = value = backend.newp(ptr)
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
