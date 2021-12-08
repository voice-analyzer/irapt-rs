use cpython::{exc, FromPyObject, NoArgs, PyClone, PyErr, PyModule, PyObject, PyResult, Python, ToPyObject};

pub struct PyPlot(pub PyModule);

#[derive(Debug)]
pub struct Figure(pub PyObject);

#[derive(Debug)]
pub struct Axes(pub PyObject);

#[derive(Debug)]
pub struct Line2D(pub PyObject);

impl PyPlot {
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let plt = PyModule::import(py, "matplotlib.pyplot")?;
        Ok(Self(plt))
    }

    pub fn figure(&self, py: Python<'_>, kwargs: impl IntoIterator<Item = (impl ToPyObject, impl ToPyObject)>) -> PyResult<Figure> {
        let kwargs = super::python::kwargs(py, kwargs)?;
        let figure = self.0.call(py, "figure", NoArgs, kwargs.as_ref())?;
        Ok(Figure(figure))
    }

    pub fn show(&self, py: Python<'_>) -> PyResult<()> {
        let _ignore = self.0.call(py, "show", NoArgs, None);
        Ok(())
    }
}

impl Figure {
    pub fn animation<F, KW, K, V>(&self, py: Python<'_>, func: F, kwargs: KW) -> PyResult<PyObject>
    where
        F: ToPyObject,
        KW: IntoIterator<Item = (K, V)>,
        K: ToPyObject,
        V: ToPyObject,
    {
        let animation_module = PyModule::import(py, "matplotlib.animation")?;
        let kwargs = super::python::kwargs(py, kwargs)?;
        let animation = animation_module.call(py, "FuncAnimation", (&self.0, func), kwargs.as_ref())?;
        Ok(animation)
    }

    py_methods! {
        pub fn add_axes(rect) -> Axes;
    }
}

impl_to_from_py_object!(Figure, "matplotlib.pyplot");

impl Axes {
    py_methods! {
        pub fn grid();
        pub fn plot(*) -> Vec<Line2D>;
        pub fn set_xlim(min, max);
        pub fn set_ylim(min, max);
    }
}

impl_to_from_py_object!(Axes, "matplotlib.pyplot");

impl Line2D {
    py_methods! {
        pub fn set_data(x_data, y_data);
    }
}

impl_to_from_py_object!(Line2D, "matplotlib.lines");
