#[macro_use]
mod python;

mod pyplot;

use cpython::{PyClone, PyList, Python, PythonObject, ToPyObject};
use std::cell::RefCell;
use std::convert::TryInto;
use std::sync::Arc;

use self::pyplot::{Line2D, PyPlot};
use self::python::{NoKwArgs, PyContext, PyFnMut, PyListExt as _};

use super::DisplayData;

pub fn run(display: Arc<DisplayData>) -> anyhow::Result<()> {
    let python_lock = Python::acquire_gil();
    let py = python_lock.python();
    let plt = PyPlot::new(py).py_context("the matplotlib python library is required but could not be loaded")?;

    let max_pitches = 60;
    let max_pitch = 500.0;

    let figure = plt.figure(py, NoKwArgs).py_caller_context()?;
    let pitches_axes = figure.add_axes(py, &[0, 0, 1, 1][..], NoKwArgs).py_caller_context()?;
    pitches_axes.grid(py, NoKwArgs).py_caller_context()?;
    pitches_axes.set_xlim(py, 0, max_pitches, NoKwArgs).py_caller_context()?;
    pitches_axes.set_ylim(py, 0.0, max_pitch, NoKwArgs).py_caller_context()?;

    let x_data = PyList::new(py, &[]);
    let y_data = PyList::new(py, &[]);
    let pitches_lines = pitches_axes
        .plot(py, (x_data.clone_ref(py), y_data.clone_ref(py)), NoKwArgs)
        .py_caller_context()?;
    let [pitches_line]: [Line2D; 1] = pitches_lines.try_into().unwrap();

    type AnimationContext = (Line2D, PyList, PyList);
    let animation_context: AnimationContext = (pitches_line, x_data, y_data);
    let animation_callback = PyFnMut::create_instance(
        py,
        RefCell::new(Box::new(move |py, args, _kwargs| {
            let (_frame, (pitches_line, x_data, y_data)): (u64, AnimationContext) = args.as_object().extract(py)?;
            if let Some(new_pitches) = display.pitches.take() {
                let mut tentative_pitches = Vec::from(new_pitches);
                if let Some(new_pitch_frequency) = tentative_pitches.pop().and_then(|pitch| pitch.frequency) {
                    if x_data.len(py) < max_pitches {
                        x_data.append(py, x_data.len(py).into_py_object(py).into_object());
                    } else {
                        y_data.pop(py, 0)?;
                    }
                    y_data.append(py, new_pitch_frequency.into_py_object(py).into_object());
                    pitches_line.set_data(py, x_data, y_data, NoKwArgs)?;
                }
            }
            Ok((pitches_line.0,).into_py_object(py).into_object())
        })),
    )
    .py_caller_context()?;
    let animation = figure
        .animation(py, animation_callback, [
            ("fargs", (animation_context,).into_py_object(py).into_object()),
            ("interval", 30u32.into_py_object(py).into_object()),
        ])
        .py_caller_context()?;

    plt.show(py).py_caller_context()?;
    drop(animation);

    Ok(())
}
