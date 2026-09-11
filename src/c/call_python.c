static PyObject *_current_interp_key(void)
{
    PyInterpreterState *interp = PyInterpreterState_Get();
    return PyInterpreterState_GetDict(interp);   /* shared reference */
}

static PyObject *_get_interpstate_dict(void)
{
    /* Hack around to return a dict that is subinterpreter-local.
       Does not return a new reference.  Returns NULL in case of
       error, but without setting any exception.  (If called late
       during shutdown, we *can't* set an exception!)
    */
    static PyObject *attr_name = NULL;
    PyThreadState *tstate;
    PyObject *d, *interpdict;
    PyInterpreterState *interp;

#if PY_VERSION_HEX >= 0x030D0000
    tstate = PyThreadState_GetUnchecked();
#else
    tstate = _PyThreadState_UncheckedGet();
#endif
    if (tstate == NULL) {
        /* no thread state! */
        return NULL;
    }

    interp = PyThreadState_GetInterpreter(tstate);
    interpdict = PyInterpreterState_GetDict(interp);   /* shared reference */
    if (interpdict == NULL) {
        /* subinterpreter was cleared already, or is being cleared right now,
           to a point that is too much for us to continue */
        return NULL;
    }

    /* from there on, we know the (sub-)interpreter is still valid */

    if (attr_name == NULL) {
        attr_name = PyUnicode_InternFromString("__cffi_backend_extern_py");
        if (attr_name == NULL)
            goto error;
    }

    d = PyDict_GetItem(interpdict, attr_name);
    if (d == NULL) {
        PyObject *candidate = PyDict_New();
        if (candidate == NULL)
            goto error;
        /* Allocation may reenter registration; preserve its registry. */
        d = PyDict_SetDefault(interpdict, attr_name, candidate);
        Py_DECREF(candidate);
        if (d == NULL)
            goto error;
    }
    return d;

 error:
    PyErr_Clear();    /* typically a MemoryError */
    return NULL;
}

static PyObject *_ffi_def_extern_decorator_lock_held(PyObject *outer_args,
                                                    PyObject *fn)
{
    const char *s;
    PyObject *error, *onerror, *infotuple, *old1;
    int index, err;
    const struct _cffi_global_s *g;
    struct _cffi_externpy_s *externpy;
    CTypeDescrObject *ct;
    FFIObject *ffi;
    builder_c_t *types_builder;
    PyObject *name = NULL;
    PyObject *interpstate_dict;
    PyObject *interpstate_key;

    if (!PyArg_ParseTuple(outer_args, "OzOO", &ffi, &s, &error, &onerror))
        return NULL;

    if (s == NULL) {
        name = PyObject_GetAttrString(fn, "__name__");
        if (name == NULL)
            return NULL;
        s = PyUnicode_AsUTF8(name);
        if (s == NULL) {
            Py_DECREF(name);
            return NULL;
        }
    }

    types_builder = &ffi->types_builder;
    index = search_in_globals(&types_builder->ctx, s, strlen(s));
    if (index < 0)
        goto not_found;
    g = &types_builder->ctx.globals[index];
    if (_CFFI_GETOP(g->type_op) != _CFFI_OP_EXTERN_PYTHON)
        goto not_found;
    Py_XDECREF(name);

    ct = realize_c_type(types_builder, types_builder->ctx.types,
                        _CFFI_GETARG(g->type_op));
    if (ct == NULL)
        return NULL;

    infotuple = prepare_callback_info_tuple(ct, fn, error, onerror, 0);
    Py_DECREF(ct);
    if (infotuple == NULL)
        return NULL;

    /* don't directly attach infotuple to externpy: in the presence of
       subinterpreters, each time we switch to a different
       subinterpreter and call the C function, it will notice the
       change and look up infotuple from the interpstate_dict.
    */
    interpstate_dict = _get_interpstate_dict();
    if (interpstate_dict == NULL) {
        Py_DECREF(infotuple);
        return PyErr_NoMemory();
    }

    externpy = (struct _cffi_externpy_s *)g->address;
    interpstate_key = PyLong_FromVoidPtr((void *)externpy);
    if (interpstate_key == NULL) {
        Py_DECREF(infotuple);
        return NULL;
    }

    err = PyDict_SetItem(interpstate_dict, interpstate_key, infotuple);
    Py_DECREF(interpstate_key);
    Py_DECREF(infotuple);    /* interpstate_dict owns the last ref */
    if (err < 0)
        return NULL;

    /* force _update_cache_to_call_python() to be called the next time
       the C function invokes cffi_call_python, to update the cache */
    old1 = cffi_atomic_load(&externpy->reserved1);
    Py_INCREF(Py_None);
    cffi_atomic_store(&externpy->reserved1, Py_None);   /* a non-NULL value */
    Py_XDECREF(old1);

    /* return the function object unmodified */
    Py_INCREF(fn);
    return fn;

 not_found:
    PyErr_Format(FFIError, "ffi.def_extern('%s'): no 'extern \"Python\"' "
                 "function with this name", s);
    Py_XDECREF(name);
    return NULL;
}

static PyObject *_ffi_def_extern_decorator(PyObject *outer_args, PyObject *fn)
{
    PyObject *result;
    CFFI_LOCK();
    result = _ffi_def_extern_decorator_lock_held(outer_args, fn);
    CFFI_UNLOCK();
    return result;
}


static int _update_cache_to_call_python(struct _cffi_externpy_s *externpy)
{
    PyObject *interpstate_dict, *interpstate_key, *infotuple, *old1, *new1;
    PyObject *old2;
    int found;

    interpstate_dict = _get_interpstate_dict();
    if (interpstate_dict == NULL)
        return 4;    /* oops, shutdown issue? */

    interpstate_key = PyLong_FromVoidPtr((void *)externpy);
    if (interpstate_key == NULL)
        goto error;

    found = PyDict_GetItemRef(interpstate_dict, interpstate_key, &infotuple);
    Py_DECREF(interpstate_key);
    if (found < 0)
        goto error;
    if (!found)
        return 3;    /* no ffi.def_extern() from this subinterpreter */

    new1 = _current_interp_key();
    if (new1 == NULL) {
        Py_DECREF(infotuple);
        goto error;
    }
    Py_INCREF(new1);
    old1 = (PyObject *)cffi_atomic_load(&externpy->reserved1);
    old2 = (PyObject *)externpy->reserved2;
    externpy->reserved2 = infotuple;    /* takes ownership */
    cffi_atomic_store(&externpy->reserved1, new1);  /* holds a reference */
    Py_XDECREF(old1);
    Py_XDECREF(old2);

    return 0;   /* no error */

 error:
    PyErr_Clear();
    return 2;   /* out of memory? */
}

#if (defined(WITH_THREAD) && !defined(_MSC_VER) &&   \
     !defined(__amd64__) && !defined(__x86_64__) &&   \
     !defined(__i386__) && !defined(__i386))
# if defined(HAVE_SYNC_SYNCHRONIZE)
#   define read_barrier()  __sync_synchronize()
# elif defined(_AIX)
#   define read_barrier()  __lwsync()
# elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#   include <mbarrier.h>
#   define read_barrier()  __compiler_barrier()
# elif defined(__hpux)
#   define read_barrier()  _Asm_mf()
# else
#   define read_barrier()  /* missing */
#   warning "no definition for read_barrier(), missing synchronization for\
 multi-thread initialization in embedded mode"
# endif
#else
# define read_barrier()  (void)0
#endif

static void cffi_call_python(struct _cffi_externpy_s *externpy, char *args)
{
    /* Invoked by the helpers generated from extern "Python" in the cdef.

       'externpy' is a static structure that describes which of the
       extern "Python" functions is called.  It has got fields 'name' and
       'type_index' describing the function, and more reserved fields
       that are initially zero.  These reserved fields are set up by
       ffi.def_extern(), which invokes _ffi_def_extern_decorator() above.

       'args' is a pointer to an array of 8-byte entries.  Each entry
       contains an argument.  If an argument is less than 8 bytes, only
       the part at the beginning of the entry is initialized.  If an
       argument is 'long double' or a struct/union, then it is passed
       by reference.

       'args' is also used as the place to write the result to
       (directly, even if more than 8 bytes).  In all cases, 'args' is
       at least 8 bytes in size.
    */
    int err = 0;

    /* Generated embedding modules may publish initialization with a write
       barrier.  Pair it before accessing interpreter and module state. */
    read_barrier();

    save_errno();

    /* We need the infotuple here.  We could always go through
       _update_cache_to_call_python(), but to avoid the extra dict
       lookups, we cache in (reserved1, reserved2) the last seen pair
       (interp->modules, infotuple).  The first item in this tuple is
       a random PyObject that identifies the subinterpreter.
    */
    if (cffi_atomic_load(&externpy->reserved1) == NULL) {
        /* Not initialized!  We didn't call @ffi.def_extern() on this
           externpy object from any subinterpreter at all. */
        err = 1;
    }
    else {
        PyGILState_STATE state = gil_ensure();
        PyObject *infotuple = NULL;
        CFFI_LOCK();
        if (cffi_atomic_load(&externpy->reserved1) != _current_interp_key()) {
            /* Update the (reserved1, reserved2) cache.  This will fail
               if we didn't call @ffi.def_extern() in this particular
               subinterpreter. */
            err = _update_cache_to_call_python(externpy);
        }
        if (!err) {
            infotuple = (PyObject *)externpy->reserved2;
            /* reserved2 is only stable while CFFI_LOCK is held. */
            Py_INCREF(infotuple);
        }
        CFFI_UNLOCK();
        if (infotuple != NULL) {
            general_invoke_callback(0, args, args, infotuple);
        }
        gil_release(state);
    }
    if (err) {
        static const char *msg[] = {
            "no code was attached to it yet with @ffi.def_extern()",
            "got internal exception (out of memory?)",
            "@ffi.def_extern() was not called in the current subinterpreter",
            "got internal exception (shutdown issue?)",
        };
        fprintf(stderr, "extern \"Python\": function %s() called, "
                        "but %s.  Returning 0.\n", externpy->name, msg[err-1]);
        memset(args, 0, externpy->size_of_result);
    }
    restore_errno();
}
