#[macro_use]
mod macros;

use std::{cell::RefCell, fmt::Display, panic::Location};

use anyhow::anyhow;
use cpython::{py_class, PyTuple, PyDict, PyObject, PyResult, Python, ToPyObject, PyList, PythonObject, ObjectProtocol};

pub struct NoKwArgs;

py_class!(pub class PyFn |python| {
    data callback: Box<dyn Fn(Python<'_>, &PyTuple, Option<&PyDict>) -> PyResult<PyObject> + Send + 'static>;

    def __call__(&self, *args, **kwargs) -> PyResult<PyObject> {
        let callback = self.callback(python);
        callback(python, args, kwargs)
    }
});

py_class!(pub class PyFnMut |python| {
    data callback: RefCell<Box<dyn FnMut(Python<'_>, &PyTuple, Option<&PyDict>) -> PyResult<PyObject> + Send + 'static>>;

    def __call__(&self, *args, **kwargs) -> PyResult<PyObject> {
        let mut callback = self.callback(python).try_borrow_mut().expect("PyFnMut called recursively");
        callback(python, args, kwargs)
    }
});

py_class!(pub class PyFnOnce |python| {
    data callback: RefCell<Option<Box<dyn FnOnce(Python<'_>, &PyTuple, Option<&PyDict>) -> PyResult<PyObject> + Send + 'static>>>;

    def __call__(&self, *args, **kwargs) -> PyResult<PyObject> {
        let callback = self.callback(python).take().expect("PyFnOnce called more than once");
        callback(python, args, kwargs)
    }
});

pub fn kwargs<KW, K, V>(py: Python<'_>, kwargs: KW) -> PyResult<Option<PyDict>>
where KW: IntoIterator<Item = (K, V)>,
      K: ToPyObject,
      V: ToPyObject,
{
    let py_kwargs = PyDict::new(py);
    for (key, value) in kwargs {
        py_kwargs.set_item(py, key, value)?;
    }
    Ok((py_kwargs.len(py) != 0).then(|| py_kwargs))
}

impl IntoIterator for NoKwArgs {
    type Item = (PyObject, PyObject);
    type IntoIter = std::option::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        None.into_iter()
    }
}

pub trait PyListExt {
    fn pop(&self, py: Python<'_>, index: usize) -> PyResult<PyObject>;
}

impl PyListExt for PyList {
    fn pop(&self, py: Python<'_>, index: usize) -> PyResult<PyObject> {
        self.as_object().call_method(py, "pop", (index,), None)
    }
}

pub trait PyContext<T> {
    fn py_context<C>(self, context: C) -> anyhow::Result<T>
    where
        C: Display + Send + Sync + 'static;

    #[track_caller]
    fn py_caller_context(self) -> anyhow::Result<T>
    where
        Self: Sized,
    {
        self.py_context(format!("at {}", Location::caller()))
    }
}

impl<T> PyContext<T> for PyResult<T> {
    fn py_context<C>(self, context: C) -> anyhow::Result<T>
    where
        C: Display + Send + Sync + 'static,
    {
        self.map_err(|error| {
            let display_error = anyhow!(
                "python raised {}{}{}",
                error.ptype,
                error.pvalue.map(|pvalue| format!(": {}", pvalue)).unwrap_or_default(),
                error
                    .ptraceback
                    .map(|ptraceback| format!("\ntraceback:\n{}", ptraceback))
                    .unwrap_or_default(),
            );
            display_error.context(context)
        })
    }
}
