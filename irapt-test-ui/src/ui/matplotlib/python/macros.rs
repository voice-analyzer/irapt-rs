macro_rules! py_methods {
    ($($vis:vis fn $name:ident($($arg:tt),*) $(-> $return_type:ty)?);+;) => {
        $(
            py_method!($vis fn $name($($arg),*) $(-> $return_type)?);
        )*
    }
}

macro_rules! py_method {
    ($vis:vis fn $name:ident($($arg:tt),*)) => {
        py_method!($vis fn $name($($arg),*) -> PyObject);
    };

    ($vis:vis fn $name:ident() -> $return_type:ty) => {
        $vis fn $name<KW, K, V>(&self, py: Python<'_>, kwargs: KW) -> PyResult<$return_type>
        where KW: IntoIterator<Item = (K, V)>,
              K: ToPyObject,
              V: ToPyObject,
        {
            use cpython::ObjectProtocol as _;
            let kwargs = super::python::kwargs(py, kwargs)?;
            let ret = self.0.call_method(py, stringify!($name), NoArgs, kwargs.as_ref())?;
            ret.extract(py)
        }
    };

    ($vis:vis fn $name:ident(*) -> $return_type:ty) => {
        $vis fn $name<A, KW, K, V>(&self, py: Python<'_>, args: A, kwargs: KW) -> PyResult<$return_type>
        where A: cpython::ToPyObject<ObjectType = cpython::PyTuple>,
              KW: IntoIterator<Item = (K, V)>,
              K: ToPyObject,
              V: ToPyObject,
        {
            use cpython::ObjectProtocol as _;
            let kwargs = super::python::kwargs(py, kwargs)?;
            let ret = self.0.call_method(py, stringify!($name), args, kwargs.as_ref())?;
            ret.extract(py)
        }
    };

    ($vis:vis fn $name:ident($($arg:ident),+) -> $return_type:ty) => {
        $vis fn $name<KW, K, V>(&self, py: Python<'_>, $($arg: impl cpython::ToPyObject),+, kwargs: KW) -> PyResult<$return_type>
        where KW: IntoIterator<Item = (K, V)>,
              K: ToPyObject,
              V: ToPyObject,
        {
            use cpython::ObjectProtocol as _;
            let kwargs = super::python::kwargs(py, kwargs)?;
            let ret = self.0.call_method(py, stringify!($name), ($($arg),+,), kwargs.as_ref())?;
            ret.extract(py)
        }
    };
}

macro_rules! impl_to_from_py_object {
    ($ty:ty, $module:expr) => {
        impl ToPyObject for $ty {
            type ObjectType = PyObject;
            fn to_py_object(&self, py: Python<'_>) -> Self::ObjectType {
                self.0.to_py_object(py)
            }
            fn into_py_object(self, py: Python<'_>) -> Self::ObjectType {
                self.0.into_py_object(py)
            }
        }

        impl<'s> FromPyObject<'s> for $ty {
            fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self> {
                let module = PyModule::import(py, $module)?;
                let ty = module.get(py, stringify!($ty))?.cast_into(py)?;
                match obj.get_type(py).is_subtype_of(py, &ty) {
                    true => Ok(Self(obj.clone_ref(py))),
                    false => Err(PyErr::new::<exc::TypeError, _>(py, NoArgs)),
                }
            }
        }
    };
}
